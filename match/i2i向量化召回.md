[toc]

KDD 2018的Best Paper：Real-time Personalization using Embeddings for Search Ranking at Airbnb 。

## 背景

- Congregated Search: 不同于普通的搜索，用户在Airbnb的一次search session往往集中在一个区域(market)。例如，一位打算去丽江旅行的顾客，他的搜索点击行为都集中丽江这一个market，而很少会跨market。这种聚集性，会对embedding的质量造成负面影响。具体来说，此文使用negative sampling来加速embedding训练，进行负采样的时候样本基本都是跨market的，因此算法会误以为同market的都相似的，从而降低学习到的embedding的质量。
- 顾客的预定是终极目标：Airbnb的盈利是靠交易达成后抽成，因此顾客的停留时长、点击条目数、跟房主的互动次数等虽然都是正向指标，但最终极的目标还是顾客预定。算法要时刻把提升预定”放在心中“，如果仅仅提升了点击量，交易量上不去，就不是一个好算法。



## 通过用户点击数据学习Listing(房屋) Embedding

文章中切分不同的click session的依据是两次点击的时间间隔超过30min。

- 文章把一个顾客一定时间内的点击序列作为Skip-gram一条输入序列。一个顾客可能有很多次的点击，不同的点击时间跨度可能很长。文章中**切分不同的click session的依据是两次点击的时间间隔超过30min**。
- 以click session的最后顾客是否预订可以将其分为booked sessions和exploratory sessions两类。虽然这两类session对学习embedding都很重要，但booked sessions更符合我们的终极目标。为了充分利用booked sessions中的信息，文章把booked sessions中最终预订的房屋作为global context(即**Booked listing as Global Context**)。具体解释一下就是，常规skip-gram算法中使用序列中的每个词(在此即listing)通过softmax预测其周围词，加入global context之后，需要预测的词不仅包括周围词，还包括glolbal context(即booked listing)，贴一下文章的图片一看便知。

![img](https://pic4.zhimg.com/80/v2-73a10db2e097a48bf4babc9b38f95a83_1440w.jpg)

- skip-gram算法的本质是softmax分类，即把central listing的embedding作为softmax的输入，softmax输出中代表context listings的slot的值应该接近1，其余slot的值应该接近0，这是个输出空间巨大的N分类问题，N等于listing的个数。为了简化计算，文章中使用negative sampling来简化计算。由于我们说过的应用场景第三个特点，导致negative sampling方法中的正样本几乎都来自同一个market，负样本几乎都来自于其他market，会因为采样不均衡的问题降低学习到embedding的质量（举个不是十分贴切的栗子，二分类中若样本中99%都是正样本，那分类器只需要将所有的样本都判定为正就可以达到99%的准确率）。为了解决这个问题，文章中强制每次负采样的时候都从central listing所处的market内取固定数量的listing作为负样本，即**Adapting Training for Congregated Search**。
- 平台中每天都会有新的listing上架，这些listing没有与顾客的交互信息，也就学不到embedding，怎么为这些listing构造embedding呢？文章中提出根据新上listing的属性信息(例如位置、价格、房型等)从有embedding的listings中找三个属性上最相似的listing，并对这个三个listing的embedding取均值作为新上listing的embedding，也就是文章中所说的**Cold start listing embeddings**。这个方法在similarity和diversity之间找到了一个较好的平衡。

## 通过用户预订数据学习User-type和Listing-type embeddings.（订单embedding样本增强）

因为在Airbnb业务特点中提到过的业务低频特点导致的数据稀疏性，直接从顾客的预定数据学习到高质量的embedding比较困难，文章提出了一种用顾客、客房的基础信息和统计信息按照一定规则分群(即映射为User-type和Listing-type)，并对分群后的User-type和Listing-type学习embedding的方法。

以listing(客房)为例概述一下分群的方法。取Airbnb的房屋中有地区、房间数、价格这三个属性，每个属性依据一定的准则将属性值划分到不同的bucket内。例如，地区可以以城市为单位划分，价格可以分为50-150，150-300，300-500，500+这样几个bucket。对某个listing，可以依据其各个属性的值将其映射到各个属性对应的bucket中，此listing所在的bucket组合就是其listing-type(每个list-type都代表一类listing，**实际上，可以将映射规则视为决策树的决策规则，而listing-type视为决策树的一个叶子节点**)。举个栗子，一个位于深圳，价格为220一晚，房间数为1的listing A，其listing-type是”深圳_价格区间150-300_房间数1“,另外一个位于深圳，价格为270，房间数为1的listing B的listing-type 与A的listing-type相同。

![img](https://pic2.zhimg.com/80/v2-d127a7f8d30f4d3140889d3f4126c739_1440w.jpg)

另外文章也对user进行了分群，规则如下：

![img](https://pic3.zhimg.com/80/v2-6f113dcf6f166bc1079c826b402c3566_1440w.jpg)

假设一个用户U1预订序列是 ![img](https://www.zhihu.com/equation?tex=%28l_1%2Cl_2%2C...l_n%29) ,文章依据listing和user的分群规则，将预订序列变换为 ![img](https://www.zhihu.com/equation?tex=%28%28U_%7B1-user-type1%7D%2Cl_%7B1-listing-type%7D%29%2C%28U_%7B1-user-type2%7D%2Cl_%7B2-listing-type%7D%29%2C%28U_%7B1-user-typen%7D%2Cl_%7Bn-listing-type%7D%29%29)

![img](https://www.zhihu.com/equation?tex=%28%28U_%7B1-user-type1%7D%2Cl_%7B1-listing-type%7D%29%2C%28U_%7B1-user-type2%7D%2Cl_%7B2-listing-type%7D%29%2C%28U_%7B1-user-typen%7D%2Cl_%7Bn-listing-type%7D%29%29) ,序列中的每个元素变为了tuple,tuple的第一个元素是用户在预订该listing时的user-type，tuple的第二个元素是该listing的listing-type。很明显，可以看出来，用户的user-type会随着时间改变(如果不改变的话变换后的user-type序列岂不是没有增加任何信息，还是等价于原先的一个光秃秃的用户ID)，举个栗子，用户的第一个listing对应的user-type只需要从分权规则的前五条获得（此时用户没有预定任何房间，所以下面的统计信息都是没有的）。

将所有用户的预定序列(如果该用户有的话)都进行前面的变化，就得到了训练user-type和listing-type的”语料“。对于训练的过程，作者有个不能确信正确的解读，希望度过原文的读者指正。

![img](https://pic2.zhimg.com/80/v2-9ffe41d919919a42ac0a11c0392cbf55_1440w.jpg)



## In-Session Embedding构建

这部分生成的embedding主要是为了帮助通过用户近期行为去刻画对当前候选item的兴趣偏好。

**基础版**

本文session的构造是基于用户的点击行为，如果两次点击行为间的时间间隔超过30分钟，就新起一个session。embedding训练使用的是常用的skip-gram，为了加速使用negative sampling，这些都是比较常规的操作，损失函数为 ![img](https://www.zhihu.com/equation?tex=argmax_%7B%5Ctheta%7D%5Csum_%7B%28l%2Cc%29%5Cin%7BD_p%7D%7Dlog%5Cfrac%7B1%7D%7B1%2Be%5E%7B-v_c%5E%7B%27%7Dv_l%7D%7D%2B%5Csum_%7B%28l%2Cc%29%5Cin%7BD_n%7D%7Dlog%5Cfrac%7B1%7D%7B1%2Be%5E%7Bv%5E%7B%27%7D_cv_l%7D%7D)

其中 ![img](https://www.zhihu.com/equation?tex=D_p) 是session中的点击正例， ![img](https://www.zhihu.com/equation?tex=D_n) 是随机负采样中的负例

**转化行为加强版**

转化行为一直是我们做推荐的根本诉求，对于电商来讲，转化是购买，对于airbnb来说，转化行为就是下book。通过是否有转化行为将用户点击session分为booked sessions/exploratiory sessions。对于booked session，我们希望更好的优化booked item的embedding，因此我们加入了一个全局的booked item，优化session中的每个点击行为时，都要同时学习与这个booked item的关系，损失函数为 ![img](https://www.zhihu.com/equation?tex=argmax_%7B%5Ctheta%7D%5Csum_%7B%28l%2Cc%29%5Cin%7BD_p%7D%7Dlog%5Cfrac%7B1%7D%7B1%2Be%5E%7B-v_c%5E%7B%27%7Dv_l%7D%7D%2B%5Csum_%7B%28l%2Cc%29%5Cin%7BD_n%7D%7Dlog%5Cfrac%7B1%7D%7B1%2Be%5E%7Bv%5E%7B%27%7D_cv_l%7D%7D%2Blog%5Cfrac%7B1%7D%7B1%2Be%5E%7B-v%5E%7B%27%7D_%7Bl_b%7Dv_l%7D%7D) ，其中的 ![img](https://www.zhihu.com/equation?tex=v%5E%7B%27%7D_%7Bl_b%7D) 是booked item，对于exploratory session来说，仍然使用原有的损失函数。

**聚类信息加强版**

Airbnb用户的浏览行为有着很强的聚焦性，比如最近要去LA，浏览行为中的民宿应该大部分都是LA相关的，而random sample出的负例与这种分布完全相反，往往random出两个item是不同的区域，这种正例与负例分布的差别会导致在同一区域内的item间相似度学习不充分，为了解决这个问题，我们加了一个有限制条件的random sample，random sample中的item需要与center item在一个区域内，损失函数为 ![img](https://www.zhihu.com/equation?tex=argmax_%7B%5Ctheta%7D%5Csum_%7B%28l%2Cc%29%5Cin%7BD_p%7D%7Dlog%5Cfrac%7B1%7D%7B1%2Be%5E%7B-v_c%5E%7B%27%7Dv_l%7D%7D%2B%5Csum_%7B%28l%2Cc%29%5Cin%7BD_n%7D%7Dlog%5Cfrac%7B1%7D%7B1%2Be%5E%7Bv%5E%7B%27%7D_cv_l%7D%7D%2Blog%5Cfrac%7B1%7D%7B1%2Be%5E%7B-v%5E%7B%27%7D_%7Bl_b%7Dv_l%7D%7D%2B%5Csum_%7Bl%2Cm_n%5Cin%7BD_%7Bm_n%7D%7D%7Dlog%5Cfrac%7B1%7D%7B1%2Be%5E%7Bv%5E%7B%27%7D_%7Bm_n%7Dv_l%7D%7D)

**冷启动加强版**

对于推荐/搜索系统来说，冷启动一直是个需要克服的难题。对于新的item，我们倾向于根据item属性找到近似的已有item的embedding去刻画。对于airbnb来说就是找到三个地理位置最近的同等价位/属性的item的embedding取均值来表达



**Attribute-level Embedding 构建**

上述提到的方法都是用来刻画实时session内部的item从而刻画用户的偏好，用户的长期的兴趣同样很重要，比如他在NYC的booking行为和他在LA的booking行为应该有很强的一致性。对于这种长期行为刻画，最好的就是利用用户的转化行为，相比于点击行为，转化行为更能反馈出用户的根本诉求。但是基于转化行为的session会有几个问题，

- 数据过于稀疏，
- 很多人的book行为只有1个，这种是无法构成session提供学习的，相当于只有center item，没有上下文。
- 对于模型来讲，需要item至少5-10次的出现才能很好的学习到embedding，但是对于很多的房间被book的次数少于5.
- 两次book的行为中隔过长，用户的兴趣以及一些诉求可能会有改变。



上述问题的根本在于转化行为过于稀疏，为了解决数据的稀疏性，比较好的办法就是将ID类数据纬度下降一个层次变为属性类数据，我们从学习item的embedding变为学习属性(地域/价格/房间数等)的embedding，这样数据的稀疏性就下降了很多，泛化性提升。

为了解决用户兴趣转移的问题，我们通过用户的属性而不是id去刻画用户，当用户兴趣转移时他的属性应该也会改变。我们将用户属性和item属性放在一个空间中去训练，，在训练时，我们利用用户book时的属性，对于新用户/没有book行为的用户，我们采用下表中的前五个特征。

![img](https://pic3.zhimg.com/80/v2-cff8b350c40f53b595f792f765991022_1440w.jpg)

与点击行为略微不同的是，book行为需要考虑到房东的喜好，房东可以拒绝用户的book请求。我们将这种行为可以近似的考虑是房屋的属性和用户的属性不match，也就是可以理解为这是一个负例，因此模型从下图a=》b。

![img](https://pic4.zhimg.com/80/v2-5c994c27ddb10a13e4eeefca9365b187_1440w.jpg)



https://zhuanlan.zhihu.com/p/162163054

https://zhuanlan.zhihu.com/p/265974174

https://zhuanlan.zhihu.com/p/60436635

https://zhuanlan.zhihu.com/p/58542599

https://zhuanlan.zhihu.com/p/69153719

https://zhuanlan.zhihu.com/p/57313656

## 实现：

行为序列表生产

```shell
HQL="
DROP TABLE tmpr.r_sku_to_sku_click_sku_seq_fastjs_v1;
CREATE
	TABLE IF NOT EXISTS tmpr.r_sku_to_sku_click_sku_seq_fastjs_v1
	(
        uuid string, 
        ci1  string,
        seq  string
	)
    partitioned by (dt string)
    ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
    lines TERMINATED by '\n'
	location 'hdfs://ns1013/user/recsys/recpro/tmpr.db/r_sku_to_sku_click_sku_seq_fastjs_v1';

set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.parallel=true;
set hive.exec.parallel.thread.number=32;

INSERT overwrite TABLE tmpr.r_sku_to_sku_click_sku_seq_fastjs_v1 partition (dt)

SELECT
	uuid,
	cid1,
	CONCAT_WS(',', collect_set(item_sku_id)) seq,
	c.dt
FROM
(
SELECT
	a.uuid,
	a.item_sku_id,
	a.request_tm,
	a.dt
FROM
	(
		SELECT
			uuid,
			item_sku_id,
			request_tm,
			dt
		FROM
			app.bh_uuid_to_click_jsapp
		WHERE
			dt >= '2022-03-16'
			AND dt <= '2022-05-16'
			AND item_sku_id IS NOT NULL
			AND item_sku_id <> ''
			AND item_sku_id <> 'None'
			AND item_sku_id <> 'none'
			AND item_sku_id <> 'null'
		
		UNION
		
		SELECT
			uuid,
			item_sku_id,
			request_tm,
			dt
		FROM
			app.bh_uuid_to_cart_jsapp
		WHERE
			dt >= '2022-03-16'
			AND dt <= '2022-05-16'
			AND item_sku_id IS NOT NULL
			AND item_sku_id <> ''
			AND item_sku_id <> 'None'
			AND item_sku_id <> 'none'
			AND item_sku_id <> 'null'
		
		UNION
		
		SELECT
			uuid,
			item_sku_id,
			request_tm,
			dt
		FROM
			app.bh_uuid_to_ord_jsapp
		WHERE
			dt >= '2022-03-16'
			AND dt <= '2022-05-16'
			AND item_sku_id IS NOT NULL
			AND item_sku_id <> ''
			AND item_sku_id <> 'None'
			AND item_sku_id <> 'none'
			AND item_sku_id <> 'null'
		
		UNION
		
		SELECT
			uuid,
			item_sku_id,
			request_tm,
			dt
		FROM
			app.bh_uuid_to_click_app_valid_js
		WHERE
			dt >= '2022-03-16'
			AND dt <= '2022-05-16'
			AND item_sku_id IS NOT NULL
			AND item_sku_id <> ''
			AND item_sku_id <> 'None'
			AND item_sku_id <> 'none'
			AND item_sku_id <> 'null'
)a
inner join
(
SELECT
	dt,
	uuid
FROM
	(
		SELECT
			uuid,
			COUNT(DISTINCT item_sku_id) AS cnt,
			dt
		FROM
			(
				SELECT
					uuid,
					item_sku_id,
					request_tm,
					dt
				FROM
					app.bh_uuid_to_click_jsapp
				WHERE
					dt >= '2022-03-16'
					AND dt <= '2022-05-16'
				
				UNION
				
				SELECT
					uuid,
					item_sku_id,
					request_tm,
					dt
				FROM
					app.bh_uuid_to_cart_jsapp
				WHERE
					dt >= '2022-03-16'
					AND dt <= '2022-05-16'
				
				UNION
				
				SELECT
					uuid,
					item_sku_id,
					request_tm,
					dt
				FROM
					app.bh_uuid_to_ord_jsapp
				WHERE
					dt >= '2022-03-16'
					AND dt <= '2022-05-16'
				
				UNION
				
				SELECT
					uuid,
					item_sku_id,
					request_tm,
					dt
				FROM
					app.bh_uuid_to_click_app_valid_js
				WHERE
					dt >= '2022-03-16'
					AND dt <= '2022-05-16'
			)
			bb
		WHERE
			item_sku_id IS NOT NULL
			AND item_sku_id <> ''
			AND item_sku_id <> 'None'
			AND item_sku_id <> 'none'
			AND item_sku_id <> 'null'
		GROUP BY
			uuid,
			dt
	)
	bbb
	WHERE
		cnt > 3
)b
on a.uuid = b.uuid and a.dt = b.dt
ORDER BY
	uuid,
	request_tm,
	dt ASC
)c
left join
(
SELECT dt, sku, cid1 FROM app.app_search_active_sku_info WHERE dt >= '2022-03-16' and dt <= '2022-05-16' GROUP BY dt, sku, cid1
)d
on c.item_sku_id = d.sku and c.dt = d.dt
GROUP BY
	uuid,
	cid1,
	c.dt
"

HQL="
SELECT seq
FROM
(
SELECT seq, size(SPLIT(seq,',')) as len FROM tmpr.r_sku_to_sku_click_sku_seq_fastjs_v1
)a
WHERE len > 6
"
```

mapreduce（修改序列格式）

```shell
spark-submit --num-executors 500 \
             --driver-memory 30g \
             --executor-memory 30g \
             --executor-cores 4 \
             --master yarn \
             --deploy-mode cluster \
             --conf spark.sql.shuffle.partitions=100 \
             --conf spark.shuffle.service.enabled=true \
             --conf spark.yarn.maxAppAttempts=2 \
             --conf spark.executor.memoryOverhead=4196 \
             --conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
             --conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
             --conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_mart_recsys:latest \
             --conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_mart_recsys:latest \
             mapreduce_seq_trans_spark.py

if [[ $? -ne 0 ]]; then
    echo "spark score error"
    exit 1
fi
```

```python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import random
import zipfile
import sys
import commands
import numpy as np
import datetime
import tensorflow as tf
from functools import reduce
from cStringIO import StringIO
from pyspark import SparkConf
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.types import Row
from pyspark.sql.functions import udf, col

reload(sys)
sys.setdefaultencoding('utf-8')

#配置spark
conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Fastjs_sku_seq_wj").enableHiveSupport().getOrCreate()
sc = spark.sparkContext

df = spark.sql("""SELECT seq FROM tmpr.r_sku_to_sku_click_sku_seq_fastjs_tmp_v3""")

def res(line):
    return ' '.join(line.strip().split(','))

res_udf = udf(res, StringType())

df_split = df.withColumn('seq',res_udf(col("seq")))

df_split.rdd.map(lambda row: row[0]).saveAsTextFile("hdfs://ns1013/user/recsys/recpro/basic_alg/personal/cuining8/fastjs/fastjs_behavior_sku_seq_v1")
```

w2v

```python
!rm -rf ./model
!mkdir ./model
!rm -rf ./data
!mkdir ./data

import os
import sys
import logging
import multiprocessing
import numpy as np
from time import time
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath
import pydoop.hdfs as hdfs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

cores = multiprocessing.cpu_count()
logging.info("cores num:"+str(cores))

hdfs_file_path_seq = hdfs.lsl('hdfs://ns1013/user/recsys/recpro/basic_alg/personal/cuining8/fastjs/fastjs_behavior_sku_seq_v1')

class MySentences(object):
    def __init__(self, rootdir_seq):
        self.rootdir = rootdir_seq
 
    def __iter__(self):
        for path in self.rootdir:
            #print(path)
            file = path['path']
            #print(file)
            with hdfs.open(file, 'rt') as f:
                for line in f:
                    if line:
                        yield line.strip().split(' ')


sentences = MySentences(hdfs_file_path_seq)
next(sentences.__iter__())

class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        cumlative_loss = model.get_latest_training_loss()
        cum_loss_list.append(cumlative_loss )
        print('cumlative_loss after epoch {}: {}'.format(self.epoch, cumlative_loss ))
        loss_list.append(cum_loss_list[-1]-cum_loss_list[-2] if self.epoch != 1 else cum_loss_list[-1])
        print('Loss after epoch {}: {}'.format(self.epoch, loss_list[-1]))
        self.epoch += 1

def draw_loss(y):
    plt.rcParams['font.sans-serif']=['SimHei']
    x = np.arange(0,len(y))
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.title('gensim_loss')
    plt.plot(x, y,color='red')
    # plt.savefig('loss.png',dpi=120)
    plt.show()

print(gensim.__version__)


w2v_model = Word2Vec(min_count=5,
                     window=5,
                     size=32,
                     sample=0.01, 
                     alpha=0.03, 
                     min_alpha=0.0007,
                     negative=9,
                     workers=cores-1,
                     compute_loss=True,
                     sg=1,
                     ns_exponent=0.75)

t = time()
w2v_model.build_vocab(sentences, progress_per=100000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
cum_loss_list = [0]# 累计 loss
loss_list=[] #单loss
w2v_model.train(sentences=sentences, 
                total_examples=w2v_model.corpus_count, 
                epochs=15, 
                report_delay=1, 
                compute_loss=True, 
                callbacks=[callback()])
#
# train( epochs = None , start_alpha = None , end_alpha = None , word_count = 0 , queue_factor = 2 , report_delay = 1.0 , compute_loss = False , callbacks = () ,** kwargs )
#
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

print('loss_List', loss_list)
draw_loss(loss_list)


w2v_model.save("./model/fastjs_sku_emb_v1.model")


vector = w2v_model.wv['10026548850804']
print(vector)

sims = w2v_model.wv.most_similar('10026548850804', topn=10)
print(sims)

# 保存词向量
# w2v_model.wv.save("./vector/model.wv")

# Load a word2vec model stored in the C *text* format.
w2v_model.wv.save_word2vec_format('./vector/fastjs_emb_v1',binary=False)# Load a word2vec model stored in the C *binary* format.
#'/media/cfs/zhengshujian/.pylib/lib/python3.6/site-packages/gensim/test/test_data ./lbs_sku_emb'

# wv = KeyedVectors.load("./vector/model.wv", mmap='r')
# vector = wv['computer']

```

faiss

```python
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

emb_data = pd.read_csv('./vector/fastjs_emb_v1', sep='\t', names=['sku', 'emb'])
emb_data.head()

sku=[]
emb=[]
for i in range(1,len(emb_data)):
    sku.append(emb_data['sku'][i].split(' ')[0])
    emb.append(','.join([x for x in emb_data['sku'][i].split(' ')[1:]]))

emb_data_tmp = pd.DataFrame(columns=['sku','emb'])
emb_data_tmp['sku'] = sku
emb_data_tmp['emb'] = emb

emb_data_tmp.head()
emb_data_tmp.shape

feat_emb = np.zeros(shape=(emb_data_tmp.shape[0], 32))
for i in tqdm(range(emb_data_tmp.shape[0])):
    feat_emb[i] = np.array(eval(emb_data_tmp['emb'][i]))

print(feat_emb.shape)
feat_emb = feat_emb.astype('float32')
print(feat_emb.shape)

dim = 32
index_l2 = faiss.IndexFlatL2(dim)
print(index_l2.is_trained)

# 特征入库
index_l2.add(feat_emb)

t = time.time()
D, I = index_l2.search(feat_emb,100)
print('Time to search sku cost: {} mins'.format(round((time.time() - t) / 60, 2)))

from multiprocessing import Process
t = time.time()
#!rm -rf ./data/*
#global emb_data
!rm -rf ./sku2sku_data
!mkdir ./sku2sku_data
import datetime
dt =  (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")


def data_part(row_num,process_num):
    print('row_num:', row_num)
    part_row_num = row_num//process_num
    index_list = []
    start_index = 0
    for i in range(process_num):
        if i == process_num - 1:
            end_index = row_num - 1
        else:
            end_index = start_index + part_row_num
        index_list.append([int(start_index), int(end_index)])
        start_index = end_index+1
    print('part_row_nums:', index_list)
    return index_list


def get_cskus(startIndex, endIndex, process_id, file_name):
    print("process: %s\t save data start_index: %s\t end_index: %s" %(process_id, startIndex, endIndex))
    result = ''
    content = ''
    save_file = open(file_name, "w")
    for i in range(startIndex, endIndex+1):
        cskus = ''
        for j in range(1, 50):
            #cskus.append(emb_data['sku'][I[i][j]]) # 取索引查skuid
            try:
                if j == 49:
                    cskus += str(emb_data_tmp['sku'][I[i][j]]) + ':' + str(100-j)
                else:
                    cskus += str(emb_data_tmp['sku'][I[i][j]]) + ':' + str(100-j) + ',' 
            except:
                continue
#         print('cskus:',cskus, '\n')
        
        result += str(emb_data_tmp['sku'][i]) + '\t' + cskus + '\t' 
        #print('result:',result, '\n')
        
        if i == endIndex: 
            content += result
        else:
            content += result + '\n'
        result = ''
        
        if i%10000==0:
#         if i%10==0:
            #print('content:', content, '\n')
            save_file.write(content)
            content=''
                
    #print('content:', content, '\n')
    save_file.write(content)
    print('file_nums:',startIndex, endIndex)
    save_file.close()
        
          

def mult_save_data(row_num, process_num):
    part_index_list = data_part(row_num, process_num)
    process = []
    for i in range(process_num):
        file_name = './sku2sku_data/' + '%06d_0'%i
        print(file_name)
        p = Process(target = get_cskus, args = (part_index_list[i][0], part_index_list[i][1], i, file_name))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()
        

mult_save_data(emb_data.shape[0], 64)
# get_cskus(0,20, 0, './data/0')

print('Time to save sku cost: {} mins'.format(round((time.time() - t) / 60, 2)))

!hadoop fs -rm -r hdfs://ns1013/user/recsys/recpro/app.db/r_sku_to_sku_emb_fastjs_base
!hadoop fs -mkdir hdfs://ns1013/user/recsys/recpro/app.db/r_sku_to_sku_emb_fastjs_base
!hadoop fs -put ./sku2sku_data/* hdfs://ns1013/user/recsys/recpro/app.db/r_sku_to_sku_emb_fastjs_base/
```





