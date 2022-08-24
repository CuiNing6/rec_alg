









## 实现方法

u2u计算方法：

```shell
dt=$(date -d "1 day ago " "+%Y-%m-%d")

SPARK_EXECUTOR_MEMORY=32g
SPARK_EXECUTOR_CORES=6
SPARK_DRIVER_MEMORY=4g
SPARK_DRIVER_CORES=4
SPARK_EXECUTOR_DOCKER_IMAGE=bdp-docker.jd.com:5000/wise_mart_bag:latest

spark_cmd="spark-submit --num-executors 600 \
         --master yarn \
         --deploy-mode cluster \
         --conf spark.sql.crossJoin.enabled=true \
         --conf spark.sql.hive.mergeFiles=true \
         --conf spark.driver.maxResultSize=4g \
         --conf spark.sql.shuffle.partitions=1500 \
         --conf spark.executor.memory=$SPARK_EXECUTOR_MEMORY \
         --conf spark.executor.memoryOverhead=4g \
         --conf spark.executor.cores=$SPARK_EXECUTOR_CORES \
         --conf spark.driver.memory=$SPARK_DRIVER_MEMORY \
         --conf spark.driver.cores=$SPARK_DRIVER_CORES \
         --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
         --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
         --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=$SPARK_EXECUTOR_DOCKER_IMAGE \
         --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=$SPARK_EXECUTOR_DOCKER_IMAGE \
         fastjd_jaccard_ucf.py ${dt} 15"
echo "$spark_cmd"
eval "$spark_cmd"
```

```python
"""
CreatDate: 2021/8/16 下午3:48
Description: py 格式写法  user cf ，点击表计算相关用户，点击概率得分相乘倒排
"""
import datetime
import sys
import time
from functools import wraps

import numpy as np
from pyspark import SQLContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, StructType, StructField, FloatType, ArrayType

# 共线矩阵超参数，最低阈值
UV_THRESHOLD = 10  # sku被点击用户数
PV_THRESHOLD = 10  # 用户点击sku数

schema = StructType([
    StructField('uid2', StringType(), False),
    StructField('score', StringType(), False)
])


def func_time(func):
    """
    用装饰器实现函数计时
    :param func: 需要计时的函数
    :return: None
    """

    @wraps(func)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name=func.__name__))
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        # 转为int是为了去掉小数点后的数字，使输出更美观
        time_elapsed = int((t1 - t0) * 1000)
        print('[Function: {name} finished, spent time: "{time:0>8} ms"]'.format(
            name=func.__name__, time=time_elapsed))
        return result

    return function_timer


# todo: 存在返回结果为 null的问题，需要排查清楚
def top(n_rec):
    def top_n(value):
        if len(value) <= n_rec:
            return value
        index = np.argpartition(-np.array([i[1] for i in value]).astype(float), n_rec, axis=0)
        value = np.array(value)[index][:n_rec].tolist()
        return value

    return F.udf(lambda col: top_n(col), returnType=ArrayType(schema))


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


@func_time
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
    dt_start = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=n_days)).strftime('%Y-%m-%d')
    df = spark.table(table).filter(F.col('dt') >= dt_start).selectExpr('item_sku_id as sku', 'uuid as uid').distinct()
    print_df(df, '未过滤前')
    if m_type == 'click':
        # 映射至spu进行处理，通过spu构建共现关系
        dt_yesterday = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        df_spu = spark.table('recall.m_sku_information').selectExpr('item_sku_id as sku', 'main_sku_id as spu')
        df = df.join(df_spu, 'sku').filter('spu is not null').selectExpr('uid', 'spu as sku').distinct()
    df_user = df.groupBy('uid').agg(F.countDistinct('sku').alias('cnt')).filter('cnt > %d' % PV_THRESHOLD)
    df_sku = df.groupBy('sku').agg(F.countDistinct('uid').alias('cnt'))
    n = df_sku.approxQuantile('cnt', [0.999, 0.9995], 0.0001)[0]
    df_sku = df_sku.filter('cnt > %d' % UV_THRESHOLD).filter('cnt < %d' % n)
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


@func_time
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


@func_time
def jaccard_recommend(spark, df_master, df_slave, table, dt, n_rec=100,
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
    df = df_master.withColumnRenamed('uid', 'uid1').join(df_slave.withColumnRenamed('uid', 'uid2'), 'sku') \
        .filter('uid1 != uid2') \
        .groupBy('uid1', 'uid2').agg(F.countDistinct('sku').alias('num')) \
        .filter(F.col('num') > co_occurrence)
    df_uid1 = df_master.groupBy('uid').agg(F.countDistinct('sku').alias('den1')).withColumnRenamed('uid', 'uid1')
    df_uid2 = df_slave.groupBy('uid').agg(F.countDistinct('sku').alias('den2')).withColumnRenamed('uid', 'uid2')

    df = df.join(df_uid1, 'uid1', 'left') \
        .join(df_uid2, 'uid2', 'left') \
        .withColumn('score', F.expr('round(num / (den1 + den2 - num),4)')) \
        .filter(F.col('score') > min_score)

    df = df.selectExpr('uid1', '(uid2,score) as score').groupby('uid1').agg(F.collect_list('score').alias('score'))

    df = df.select('uid1', top(n_rec)('score').alias('score'))

    df = df.select('uid1', F.explode('score').alias('score'))
    df = df.selectExpr('uid1', 'score.uid2 as uid2', 'score.score as score')

    n1, n2 = df.count(), df.select('uid1').distinct().count()
    print('共向 %d 个用户推荐 %d 个用户，平均推荐数为:%f' % (n2, n1, n1 / n2))
    # 用户与用户的关系表
    save_as_table(spark, df.withColumn('dt', F.lit(dt)).repartition(1), table)


def main():
    dt = sys.argv[1]
    print('计算日期:%s' % dt)
    n_days = int(sys.argv[2])

    spark = (SparkSession
             .builder
             .appName("fastjd_jaccard_ucf")
             .enableHiveSupport()
             .getOrCreate())
    # 1. 点击
    df_click = get_matrix(spark, dt, n_days, 'click')
    df_ord = get_matrix(spark, dt, n_days, 'ord')

    # 2. 计算存表
    print('===============================订单相关词表===============================')
    jaccard_recommend(spark, df_ord, df_ord, 'app.fastjd_jaccard_ord_user', dt, n_rec=20)
    print('===============================点击相关词表===============================')
    jaccard_recommend(spark, df_click, df_click, 'app.fastjd_jaccard_click_user', dt, n_rec=20)
    print('程序运行成功')


if __name__ == "__main__":
    main()

```

u2i计算方法：

```shell
dt=$(date -d "1 day ago " "+%Y-%m-%d")

SPARK_EXECUTOR_MEMORY=32g
SPARK_EXECUTOR_CORES=6
SPARK_DRIVER_MEMORY=4g
SPARK_DRIVER_CORES=4
SPARK_EXECUTOR_DOCKER_IMAGE=bdp-docker.jd.com:5000/wise_mart_bag:latest

spark_cmd="spark-submit --num-executors 600 \
         --master yarn \
         --deploy-mode cluster \
         --conf spark.sql.crossJoin.enabled=true \
         --conf spark.sql.hive.mergeFiles=true \
         --conf spark.driver.maxResultSize=4g \
         --conf spark.sql.shuffle.partitions=1500 \
         --conf spark.executor.memory=$SPARK_EXECUTOR_MEMORY \
         --conf spark.executor.memoryOverhead=4g \
         --conf spark.executor.cores=$SPARK_EXECUTOR_CORES \
         --conf spark.driver.memory=$SPARK_DRIVER_MEMORY \
         --conf spark.driver.cores=$SPARK_DRIVER_CORES \
         --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
         --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
         --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=$SPARK_EXECUTOR_DOCKER_IMAGE \
         --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=$SPARK_EXECUTOR_DOCKER_IMAGE \
         fastjd_jaccard_ucf2.py ${dt}"
echo "$spark_cmd"
eval "$spark_cmd"
```

```python
"""
CreatDate: 2021/8/16 下午3:48
Description: py 格式写法  user cf ，点击表计算相关用户，点击概率得分相乘倒排
"""
import datetime
import sys
import time
from functools import wraps

import numpy as np
from pyspark import SQLContext
from pyspark.sql import SparkSession, functions as F, DataFrame, Window
from pyspark.sql.types import StringType, BooleanType, ArrayType, StructField, StructType

click_schema = StructType([
    StructField('sku', StringType(), False),
    StructField('prob', StringType(), False)
])


def func_time(func):
    """
    用装饰器实现函数计时
    :param func: 需要计时的函数
    :return: None
    """

    @wraps(func)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name=func.__name__))
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        # 转为int是为了去掉小数点后的数字，使输出更美观
        time_elapsed = int((t1 - t0) * 1000)
        print('[Function: {name} finished, spent time: "{time:0>8} ms"]'.format(
            name=func.__name__, time=time_elapsed))
        return result

    return function_timer


def top_order(n_rec):
    def top_n(value):
        value = sorted(value, key=lambda x: x.score, reverse=True)
        res = []
        for i in value:
            res.extend(zip([i.score * 100] * len(i.order_list), i.order_list))
        return res[:n_rec]

    return F.udf(lambda col: top_n(col), returnType=ArrayType(click_schema))


def top(n_rec):
    def top_n(value):
        if len(value) <= n_rec:
            return value
        index = np.argpartition(-np.array([i[1] for i in value]).astype(float), n_rec, axis=0)
        value = np.array(value)[index][:n_rec].tolist()
        return value

    return F.udf(lambda col: top_n(col), returnType=ArrayType(click_schema))


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


@func_time
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


@F.udf(returnType=BooleanType())
def existing_in_array(arr, x):
    if not x:
        return False
    if not arr:
        return True
    if x in set(arr):
        return False
    return True


@func_time
def get_user_click(spark, dt, n_days=30, min_prob=0.01, n_click=20):
    """
    获取用户的点击列表，用于去重，和用户喜欢点击的n_click个商品
    :param spark:
    :param dt:
    :param n_days:
    :param min_prob:
    :param n_click:
    :return:
    """
    dt_start = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=n_days)).strftime('%Y-%m-%d')
    df_prob = spark.table('app.app_uuid_click_jisu') \
        .filter(F.col('dt') >= dt_start) \
        .selectExpr('uuid as uid', 'item_sku_id as sku', 'prob') \
        .groupby('uid', 'sku').agg(F.max('prob').alias('prob')) \
        .filter('sku is not null')
    df_set = df_prob.groupBy('uid').agg(F.collect_set('sku').alias('sku_set'))
    print_df(df_set, '用户点击列表')
    # 保留n_rec 个sku
    # 映射至spu进行处理，通过spu构建共现关系
    dt_yesterday = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    df_spu = spark.table('recall.m_sku_information').selectExpr('item_sku_id as sku', 'main_sku_id as spu', 'item_third_cate_cd as cid3')
    df = df_prob.join(df_spu, 'sku').selectExpr('uid', 'sku', 'cid3', 'spu', 'prob')
    df = df.filter('prob > %f and prob is not null' % min_prob) \
        .selectExpr('uid', 'spu', 'cid3', '(sku,prob) as score') \
        .groupby('uid', 'spu', 'cid3') \
        .agg(F.collect_list('score').alias('score')) \
        .select('uid', 'spu', 'cid3', top(n_click)('score').alias('score')) \
        .select('uid', 'spu', 'cid3', F.explode('score').alias('score')) \
        .selectExpr('uid', 'spu', 'cid3', 'score.sku as sku', 'score.prob as click_prob')
    # 同spu 保留一个，同c3保留3个
    df = df.select('uid', 'sku', 'spu', 'cid3', 'click_prob',
                   F.row_number().over(Window.partitionBy('uid', 'spu').orderBy(F.col('click_prob').desc())).alias(
                       'rank1'),
                   F.row_number().over(Window.partitionBy('uid', 'cid3').orderBy(F.col('click_prob').desc())).alias(
                       'rank2')) \
        .filter('rank1 < 2 and rank2 < 8')
    df = df.select('uid', 'spu', 'cid3', 'sku', 'click_prob')
    print_df(df, '用户点击top商品')
    return df, df_set


@func_time
def get_user_order(spark, dt, n_days, min_order=5, max_order=100) -> DataFrame:
    """
    获取用户历史订单行为
    :param spark:
    :param dt:
    :param n_days:
    :param min_order:
    :param max_order:
    :return:
    """
    dt = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=n_days)).strftime('%Y-%m-%d')
    df = spark.table('app.bh_uuid_to_ord_jsapp') \
        .filter(F.col('dt') >= dt) \
        .groupby('uuid') \
        .agg(F.collect_set('item_sku_id').alias('order_list'))
    print_df(df, f'取{n_days}天用户点击数据共计')
    df = df.filter(F.size('order_list') > min_order) \
        .filter(F.size('order_list') < max_order) \
        .withColumnRenamed('uuid', 'uid')
    print_df(df, f'按照最少订单:{min_order}，最多订单:{max_order}过滤后')
    return df


@F.udf(returnType=ArrayType(StringType()))
def minus_set(set1, set2):
    if not set1 or not set2:
        return set2
    return list(set(set1).difference(set(set2)))


@func_time
def main():
    dt = sys.argv[1]
    print('计算日期:%s' % dt)

    spark = (SparkSession
             .builder
             .appName("fastjd_jaccard_ucf2")
             .enableHiveSupport()
             .getOrCreate())
    # 1. 取前一天的相似关系数据
    dt_yesterday = (datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    df_ord = spark.table('app.fastjd_jaccard_ord_user').filter(F.col('dt') == dt_yesterday)
    df_click = spark.table('app.fastjd_jaccard_click_user').filter(F.col('dt') == dt_yesterday)
    # 订单 点击 分数加权求和
    df_user = df_ord.select('uid1', 'uid2').distinct() \
        .unionByName(df_click.select('uid1', 'uid2').distinct()).distinct()
    df_user = df_user.join(df_ord.selectExpr('uid1', 'uid2', 'score as ord_score'), ['uid1', 'uid2'], 'left') \
        .join(df_click.selectExpr('uid1', 'uid2', 'score as click_score'), ['uid1', 'uid2'], 'left') \
        .na.fill({'ord_score': 0, 'click_score': 0})
    df_user = df_user.selectExpr('uid1', 'uid2', '(4 * ord_score + click_score) as score')

    # 3. 订单
    df_ord_history = get_user_order(spark, dt, 180, min_order=5, max_order=50)
    # 先处理订单
    df_rec1 = df_user.join(df_ord_history.withColumnRenamed('uid', 'uid2'), 'uid2') \
        .join(df_ord_history.selectExpr('uid as uid1', 'order_list as order_history'), 'uid1', 'left') \
        .select('uid1', 'score', minus_set('order_list', 'order_history').alias('order_list')) \
        .filter('order_list is not null')
    df_rec1 = df_rec1.selectExpr('uid1 as uid', '(score,order_list) as value') \
        .groupby('uid').agg(F.collect_list('value').alias('value')) \
        .select('uid', top_order(50)('value').alias('value'))
    df_rec1 = df_rec1.select('uid', F.explode('value').alias('value')) \
        .selectExpr('uid', 'value.sku as sku', 'value.prob as score')

    # 2. 高质量点击
    df_prob, df_click = get_user_click(spark, dt, 30, min_prob=0.01, n_click=20)
    df_prob = df_prob.withColumnRenamed('uid', 'uid2')
    df_click = df_click.withColumnRenamed('uid', 'uid1')

    df = df_user.join(df_click, 'uid1', 'left').join(df_prob, 'uid2') \
        .withColumn('rec_score', F.expr('click_prob * score')) \
        .filter(existing_in_array('sku_set', 'sku'))

    df_res = df.selectExpr('uid1', '(sku,rec_score) as value') \
        .groupby('uid1') \
        .agg(F.collect_list('value').alias('value')) \
        .select('uid1', top(100)('value').alias('value')) \
        .select('uid1', F.explode('value').alias('value')) \
        .selectExpr('uid1 as uid', 'value.sku as sku', 'value.prob as score')

    df_res = df_rec1.unionByName(df_res) \
        .select('uid', 'sku', 'score',
                F.row_number().over(Window.partitionBy('uid').orderBy(F.col('score').desc())).alias('rank')) \
        .filter('rank < 100').drop('rank')
    df_res = df_res.groupby('uid', 'sku').agg(F.max('score').alias('score'))
    print_df(df_res, '推荐数据')
    n1, n2, n3 = df_res.count(), df_res.select('sku').distinct().count(), df_res.select('uid').distinct().count()
    print('共向 %d 个用户 推荐 % d 个商品，平均每个用户推荐商品 %.3f' % (n3, n2, n1 / n3))
    save_as_table(spark, df_res.withColumn('dt', F.lit(dt)).repartition(1), 'app.fastjd_jaccard_ucf_tmp')
    df_res = df_res.select('sku', F.expr('concat_ws(":",uid,cast(round(score,4) as string)) as pairs')) \
        .groupby('sku').agg(F.collect_list('pairs').alias('seed_score_pairs')) \
        .withColumnRenamed('sku', 'item') \
        .withColumn('seed_score_pairs', F.array_join('seed_score_pairs', ','))
    save_as_table(spark, df_res.withColumn('dt', F.lit(dt)).repartition(1), 'app.fastjd_jaccard_ucf')
    print('程序运行成功')


if __name__ == "__main__":
    main()

```

线上

```java
/**
 * 通用的 UuidDataFetcher fetcher，触发用户uid.
 *
 * @date 2021/8/23
 */
public class UuidDataFetcher implements IndexRecallDataFetcher {


    @Override
    public Map<String, List<String>> fetch(RecRequest request, FetchExtendConfig config) {
        List<String> uids = new ArrayList();
        String uuid = request.getBroadwayRequest().getUser().getUuid();
        if (StringUtils.isEmpty(uuid)) {
            return null;
        }
        uids.add(uuid);
        Map<String, List<String>> seedMap = Maps.newHashMap();
        seedMap.put(config.getSeedType(), uids);
        return seedMap;
    }

}
```

