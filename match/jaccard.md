









## 实现方法

```shell
dt=`date -d "1 day ago " "+%Y-%m-%d"`
spark_cmd="spark-submit --num-executors 400 \
         --driver-memory 4g \
         --executor-memory 32g \
         --executor-cores 4 \
         --master yarn \
         --deploy-mode cluster \
         --conf spark.default.parallelism=1600 \
         --conf spark.sql.hive.mergeFiles=true \
         --conf spark.sql.shuffle.partitions=1600 \
         --conf spark.yarn.maxAppAttempts=1 \
         fastjd_jaccard_cf.py ${dt} 18"
echo "$spark_cmd"
eval "$spark_cmd"
```

```python
# !/usr/bin/python3
# -*-coding:utf-8-*-
"""
Author: 邓剑
CreatDate: 2021/8/4 下午3:48
Description: py 格式写法
"""
import datetime
import os
import sys

from pyspark import SQLContext
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import StringType

# 共线矩阵超参数，最低阈值
UV_THRESHOLD = 10  # sku被点击用户数
PV_THRESHOLD = 3  # 用户点击sku数
os.environ['PYSPARK_PYTHON'] = "/usr/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = "/usr/bin/python"


def print_df(df, describe, n_show=5):
    """
    打印输出spark DataFrame
    :param df:
    :param describe:
    :param n_show:
    :return:
    """
    df.cache()
    cnt = df.count()
    assert cnt > 0, '数据为空'
    print('%s,数量量:%d, partitions: %d' % (describe, cnt, df.rdd.getNumPartitions()))
    print(df)
    df.show(n=n_show)


def get_matrix(spark, dt, n_days, m_type):
    """
    获取点击或订单的共现矩阵
    :param spark:
    :param dt:
    :param n_days:
    :param m_type:
    :return:
    """
    assert m_type in ('click', 'ord'), '输入类型:%s 无该类型数据存在' % m_type
    table = "app.bh_uuid_to_click_jsapp" if m_type == 'click' else 'app.bh_uuid_to_ord_jsapp'
    # 订单时间周期放长
    n_days = 10 * n_days if m_type == 'ord' else n_days
    dt = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=n_days)).strftime('%Y-%m-%d')
    df = spark.table(table).filter(F.col('dt') >= dt).selectExpr('item_sku_id as sku', 'uuid as uid').distinct()
    print_df(df, '未过滤前')
    df_user = df.groupBy('uid').agg(F.countDistinct('sku').alias('cnt')).filter('cnt > %d' % PV_THRESHOLD)
    df_sku = df.groupBy('sku').agg(F.countDistinct('uid').alias('cnt')).filter('cnt > %d' % UV_THRESHOLD)
    df = df.join(df_user.select('uid'), 'uid').join(df_sku.select('sku'), 'sku')
    print_df(df, '%s 过滤后的共线矩阵数据' % m_type)
    return df


def existing(spark, table):
    """
    判断表是否存在
    :param spark: SparkSession
    :param table : db.tb
    :return:
    """
    assert len(table.split('.')) == 2, '表名输入有误，正确格式为 db.tb'
    db, tb = table.split('.')
    return tb in SQLContext(spark.sparkContext).tableNames(db)


def save_as_table(spark, df, table, part_col='dt'):
    """

    :param spark:
    :param df:
    :param table:
    :param part_col:
    :return:
    """
    if existing(spark, table):
        print('%s 表已存在' % table)
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        df.write.mode('overwrite').format("parquet").insertInto(table, True)
    else:
        print('%s 表不存在' % table)
        df.write.partitionBy(part_col).saveAsTable(table)
    print('写入hive表： %s成功' % table)


@F.udf(returnType=StringType())
def concat_rec(item, score):
    return '%s:%.4f' % (item, score)


def jaccard_recommend(spark, df_master, df_slave, table, dt, n_rec=50,
                      co_occurrence=3, min_score=1e-4):
    """
    协同过滤
    :param spark:
    :param df_master:
    :param df_slave:
    :param table:
    :param n_rec:
    :param dt:
    :param co_occurrence:
    :param min_score:
    :return:
    """
    # 计算相似分
    df = df_master.withColumnRenamed('sku', 'sku1').join(df_slave.withColumnRenamed('sku', 'sku2'), 'uid') \
        .filter('sku1 != sku2') \
        .groupBy('sku1', 'sku2').agg(F.countDistinct('uid').alias('num')) \
        .filter(F.col('num') > co_occurrence)
    df_sku1 = df_master.groupBy('sku').agg(F.countDistinct('uid').alias('den1')).withColumnRenamed('sku', 'sku1')
    df_sku2 = df_slave.groupBy('sku').agg(F.countDistinct('uid').alias('den2')).withColumnRenamed('sku', 'sku2')

    df = df.join(df_sku1, 'sku1', 'left') \
        .join(df_sku2, 'sku2', 'left') \
        .withColumn('score', F.expr('round(num / (den1 + den2 - num),4)')) \
        .filter(F.col('score') > min_score)
    df = df.select('sku1', 'sku2', 'score',
                   F.row_number().over(Window.partitionBy('sku1').orderBy(F.col('score').desc())).alias('rank')) \
        .filter(F.col('rank') < n_rec)

    n1, n2 = df.count(), df.select('sku1').distinct().count()
    print('共向 %d 个商品推荐 %d 个商品，平均推荐数为:%f' % (n2, n1, n1 / n2))
    # 转换格式并存表
    # 原生表
    save_as_table(spark, df.withColumn('dt', F.lit(dt)).repartition(10), table + '_tmp')
    df = df.withColumn('rec', concat_rec('sku2', 'score'))
    df = df.groupBy('sku1').agg(F.collect_list('rec').alias('rec')) \
        .withColumnRenamed('sku1', 'item').withColumn('seed_score_pairs', F.array_join('rec', ','))
    print_df(df, '最终存表数据')
    save_as_table(spark, df.select('item', 'seed_score_pairs').withColumn('dt', F.lit(dt)).repartition(10), table)


def main():
    dt = sys.argv[1]
    print('计算日期:%s' % dt)
    n_days = int(sys.argv[2])

    spark = (SparkSession
             .builder
             .appName("fastjd_jaccard_cf")
             .enableHiveSupport()
             .config("spark.executor.instances", "50")
             .config("spark.executor.memory", "4g")
             .config("spark.executor.cores", "2")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "500")
             .config("spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")
             .config("spark.executorEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")
             .config("spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name",
                     "bdp-docker.jd.com:5000/wise_mart_bag:latest")
             .config("spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name",
                     "bdp-docker.jd.com:5000/wise_mart_bag:latest")
             .getOrCreate())
    # 1. 点击
    df_click = get_matrix(spark, dt, n_days, 'click')
    df_ord = get_matrix(spark, dt, n_days, 'ord')

    # 2. 计算存表
    print('===============================订单相关词表===============================')
    jaccard_recommend(spark, df_ord, df_ord, 'app.fastjd_jaccard_ord_sku', dt)
    print('===============================点击相关词表===============================')
    jaccard_recommend(spark, df_click, df_click, 'app.fastjd_jaccard_click_sku', dt)
    print('程序运行成功')


if __name__ == "__main__":
    main()

```

