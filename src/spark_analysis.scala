// Spark 银行营销分析
// 组件：SparkCore, SparkSQL, SparkMLlib
// 目标：预测定期存款订阅（y）
//
// 运行说明：请在【仓库根目录】下启动 spark-shell 并加载本脚本，例如
//     spark-shell -i src/spark_analysis.scala
// 数据统一从 data/ 读取，结果统一写入 output/。

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline

println("=" * 60)
println("Bank Marketing Analysis with Spark")
println("=" * 60)

// 初始化 SparkSession
val spark = SparkSession.builder()
  .appName("BankMarketing")
  .master("local[*]")
  .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
println("Spark version: " + spark.version)

// ============================================================
// 1. 加载数据
// ============================================================
// 读取 CSV 文件，指定分隔符为分号，自动推断数据类型
val df = spark.read.option("header", "true").option("inferSchema", "true").option("sep", ";").csv("data/bank.csv")
println(s"Data: ${df.count()} rows x ${df.columns.length} columns")
df.createOrReplaceTempView("bank")  // 注册临时视图供 SQL 使用

// ============================================================
// 2. 使用 SparkSQL 进行数据分析
// ============================================================
println("\n" + "=" * 60)
println("SparkSQL Data Analysis")
println("=" * 60)

// 1) 目标变量 y 的分布（订阅与未订阅）
println("\n[1] Subscription distribution:")
spark.sql("SELECT y, COUNT(*) cnt, ROUND(COUNT(*)*100.0/4521,2) pct FROM bank GROUP BY y").show()

// 2) 不同职业的订阅率
println("[2] Subscription rate by job:")
spark.sql("SELECT job, COUNT(*) total, ROUND(AVG(CASE WHEN y='yes' THEN 1 ELSE 0 END)*100,2) rate FROM bank GROUP BY job ORDER BY rate DESC").show(12)

// 3) 不同教育水平的订阅率
println("[3] Subscription rate by education:")
spark.sql("SELECT education, COUNT(*) total, ROUND(AVG(CASE WHEN y='yes' THEN 1 ELSE 0 END)*100,2) rate FROM bank GROUP BY education ORDER BY rate DESC").show()

// 4) 不同婚姻状况的订阅率
println("[4] Subscription rate by marital status:")
spark.sql("SELECT marital, COUNT(*) total, ROUND(AVG(CASE WHEN y='yes' THEN 1 ELSE 0 END)*100,2) rate FROM bank GROUP BY marital ORDER BY rate DESC").show()

// 5) 不同月份（营销活动月份）的订阅率
println("[5] Monthly campaign effect:")
spark.sql("SELECT month, COUNT(*) total, ROUND(AVG(CASE WHEN y='yes' THEN 1 ELSE 0 END)*100,2) rate FROM bank GROUP BY month ORDER BY rate DESC").show(12)

// 6) 上一次营销结果（poutcome）对订阅的影响
println("[6] Previous outcome influence:")
spark.sql("SELECT poutcome, COUNT(*) total, ROUND(AVG(CASE WHEN y='yes' THEN 1 ELSE 0 END)*100,2) rate FROM bank GROUP BY poutcome ORDER BY rate DESC").show()

// 7) 不同联系方式的订阅率
println("[7] Contact type subscription rate:")
spark.sql("SELECT contact, COUNT(*) total, ROUND(AVG(CASE WHEN y='yes' THEN 1 ELSE 0 END)*100,2) rate FROM bank GROUP BY contact ORDER BY rate DESC").show()

// 8) 按年龄段分析：人数、平均余额、订阅率
println("[8] Age group analysis:")
spark.sql("""
  SELECT CASE WHEN age<25 THEN '<25' WHEN age<35 THEN '25-35' WHEN age<45 THEN '35-45' WHEN age<55 THEN '45-55' ELSE '>55' END age_group,
    COUNT(*) total, ROUND(AVG(balance),0) avg_balance, ROUND(AVG(CASE WHEN y='yes' THEN 1 ELSE 0 END)*100,2) rate
  FROM bank GROUP BY 1 ORDER BY 1
""").show()

// 9) 订阅与未订阅客户的关键数值特征对比（年龄、余额、通话时长、联系次数等）
println("[9] Numeric comparison (yes vs no):")
spark.sql("""
  SELECT y, ROUND(AVG(age),1) avg_age, ROUND(AVG(balance),1) avg_balance,
    ROUND(AVG(duration),1) avg_duration, ROUND(AVG(campaign),1) avg_campaign
  FROM bank GROUP BY y
""").show()

// ============================================================
// 3. 特征工程
// ============================================================
println("\n" + "=" * 60)
println("SparkMLlib Modeling")
println("=" * 60)

// 类别型特征列
val catCols = Array("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
// 数值型特征列
val numCols = Array("age", "balance", "day", "duration", "campaign", "pdays", "previous")

// 构建预处理流水线阶段：
// 1) 对每个类别列使用 StringIndexer 转换为数值索引
// 2) 对标签列 y 进行 StringIndexer，生成 "label" 列
// 3) VectorAssembler 将所有特征（类别索引列 + 数值列）合并为 "features" 向量
val stages = catCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c+"_idx").setHandleInvalid("keep")) ++
  Array(
    new StringIndexer().setInputCol("y").setOutputCol("label"),
    new VectorAssembler().setInputCols(catCols.map(_+"_idx") ++ numCols).setOutputCol("features")
  )

val preprocessPipeline = new Pipeline().setStages(stages)

// 划分训练集（70%）和测试集（30%）
val Array(trainDF, testDF) = df.randomSplit(Array(0.7, 0.3), seed = 42)
println(s"Train: ${trainDF.count()}, Test: ${testDF.count()}")

// 在训练集上拟合预处理流水线，并转换训练集和测试集
val preprocessorModel = preprocessPipeline.fit(trainDF)
val trainData = preprocessorModel.transform(trainDF).cache()
val testData = preprocessorModel.transform(testDF).cache()

// ============================================================
// 4. 逻辑回归模型
// ============================================================
println("\n--- Logistic Regression ---")
val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(50)
val lrModel = lr.fit(trainData)
val lrPred = lrModel.transform(testData)

// 使用 AUC（曲线下面积）评估二分类模型性能
val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction")
val lrAUC = evaluator.evaluate(lrPred)
println(f"LR AUC: $lrAUC%.4f")

// 输出逻辑回归系数（特征重要性）
val cols = catCols.map(_+"_idx") ++ numCols
println("Top 10 coefficients:")
cols.zip(lrModel.coefficients.toArray).sortBy(-_._2.abs).take(10).foreach{case (k,v) => println(f"  $k%-20s: $v%8.4f")}

// ============================================================
// 5. 随机森林模型
// ============================================================
println("\n--- Random Forest ---")
val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(30).setMaxDepth(8).setSeed(42)
val rfModel = rf.fit(trainData)
val rfPred = rfModel.transform(testData)
val rfAUC = evaluator.evaluate(rfPred)
println(f"RF AUC: $rfAUC%.4f")

// 输出随机森林的特征重要性
println("Top 10 feature importance:")
cols.zip(rfModel.featureImportances.toArray).sortBy(-_._2).take(10).foreach{case (k,v) => println(f"  $k%-20s: $v%8.6f")}

// ============================================================
// 6. 决策树模型
// ============================================================
println("\n--- Decision Tree ---")
val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxDepth(8).setSeed(42)
val dtModel = dt.fit(trainData)
val dtPred = dtModel.transform(testData)
val dtAUC = evaluator.evaluate(dtPred)
println(f"DT AUC: $dtAUC%.4f")

// ============================================================
// 7. 模型对比
// ============================================================
println("\n" + "=" * 60)
println("Model Comparison")
println("-" * 50)
println(f" Model          AUC")
println(f" Logistic Reg.  $lrAUC%.4f")
println(f" Random Forest  $rfAUC%.4f")
println(f" Decision Tree  $dtAUC%.4f")
println("-" * 50)
val best = if(rfAUC >= lrAUC && rfAUC >= dtAUC) "Random Forest" else if(lrAUC >= dtAUC) "Logistic Regression" else "Decision Tree"
println(s"Best model: $best")

// ============================================================
// 8. 保存评估结果到 JSON 文件
// ============================================================
import java.io.{File, PrintWriter}

val json = s"""{"lr_auc": $lrAUC, "rf_auc": $rfAUC, "dt_auc": $dtAUC, "best": "$best", "train_n": ${trainDF.count()}, "test_n": ${testDF.count()}}"""
new File("output").mkdirs()  // 确保输出目录存在
val pw = new PrintWriter(new File("output/model_results.json"))
pw.write(json); pw.close()
println("\nResults saved to output/model_results.json")

// 关闭 Spark 会话
spark.stop()
println("\nAnalysis complete!")
System.exit(0)
