[toc]

node2vec是一种综合考虑DFS邻域和BFS邻域的graph embedding方法。简单来说，可以看作是deepwalk的一种扩展，是结合了DFS和BFS随机游走的deepwalk。

## 顶点序列采样策略

node2vec依然采用随机游走的方式获取顶点的近邻序列，不同的是node2vec采用的是一种有偏的随机游走。

给定当前顶点 ![img](https://www.zhihu.com/equation?tex=v) ，访问下一个顶点 ![img](https://www.zhihu.com/equation?tex=x) 的概率为

![img](https://pic2.zhimg.com/80/v2-84cc0b66ec34043f82649f0d799997e1_1440w.jpg)

![img](https://www.zhihu.com/equation?tex=%5Cpi_%7Bvx%7D) 是顶点 ![img](https://www.zhihu.com/equation?tex=v) 和顶点 ![img](https://www.zhihu.com/equation?tex=x) 之间的未归一化转移概率， ![img](https://www.zhihu.com/equation?tex=Z) 是归一化常数。

node2vec引入两个超参数 ![img](https://www.zhihu.com/equation?tex=p) 和 ![img](https://www.zhihu.com/equation?tex=q) 来控制随机游走的策略，假设当前随机游走经过边 ![img](https://www.zhihu.com/equation?tex=%28t%2Cv%29) 到达顶点 ![img](https://www.zhihu.com/equation?tex=v) 设 ![img](https://www.zhihu.com/equation?tex=%5Cpi_%7Bvx%7D%3D%5Calpha_%7Bpq%7D%28t%2Cx%29%5Ccdot+w_%7Bvx%7D) ， ![img](https://www.zhihu.com/equation?tex=w_%7Bvx%7D) 是顶点 ![img](https://www.zhihu.com/equation?tex=v) 和 ![img](https://www.zhihu.com/equation?tex=x) 之间的边权，

![img](https://pic3.zhimg.com/80/v2-0d170e5c120681823ed6880411a0478e_1440w.jpg)

![img](https://www.zhihu.com/equation?tex=d_%7Btx%7D) 为顶点 ![img](https://www.zhihu.com/equation?tex=t) 和顶点 ![img](https://www.zhihu.com/equation?tex=x) 之间的最短路径距离。

下面讨论超参数 ![img](https://www.zhihu.com/equation?tex=p) 和 ![img](https://www.zhihu.com/equation?tex=q) 对游走策略的影响

- Return parameter,p

参数![img](https://www.zhihu.com/equation?tex=p)控制重复访问刚刚访问过的顶点的概率。 注意到![img](https://www.zhihu.com/equation?tex=p)仅作用于 ![img](https://www.zhihu.com/equation?tex=d_%7Btx%7D%3D0) 的情况，而 ![img](https://www.zhihu.com/equation?tex=d_%7Btx%7D%3D0) 表示顶点 ![img](https://www.zhihu.com/equation?tex=x) 就是访问当前顶点 ![img](https://www.zhihu.com/equation?tex=v) 之前刚刚访问过的顶点。 那么若 ![img](https://www.zhihu.com/equation?tex=p) 较高，则访问刚刚访问过的顶点的概率会变低，反之变高。

- In-out papameter,q

![img](https://www.zhihu.com/equation?tex=q) 控制着游走是向外还是向内，若 ![img](https://www.zhihu.com/equation?tex=q%3E1) ，随机游走倾向于访问和 ![img](https://www.zhihu.com/equation?tex=t) 接近的顶点(偏向BFS)。若 ![img](https://www.zhihu.com/equation?tex=q%3C1) ，倾向于访问远离 ![img](https://www.zhihu.com/equation?tex=t) 的顶点(偏向DFS)。

下面的图描述的是当从 ![img](https://www.zhihu.com/equation?tex=t) 访问到 ![img](https://www.zhihu.com/equation?tex=v) 时，决定下一个访问顶点时每个顶点对应的 ![img](https://www.zhihu.com/equation?tex=%5Calpha) 。

![img](https://pic3.zhimg.com/80/v2-a4a45ea71c00a8d725916dcea50f9cea_1440w.jpg)



## 学习算法

采样完顶点序列后，剩下的步骤就和deepwalk一样了，用word2vec去学习顶点的embedding向量。 值得注意的是node2vecWalk中不再是随机抽取邻接点，而是按概率抽取，node2vec采用了Alias算法进行顶点采样。

https://zhuanlan.zhihu.com/p/54867139

https://github.com/shenweichen/GraphEmbedding/blob/master/ge/walker.py



## 实现方法

共点击表生产

```shell
io.sh
##### 运行环境参数
setcof="
set hive.optimize.skewjoin = true;
set hive.cli.print.header=true;
set mapreduce.input.fileinputformat.split.maxsize=2147483648;
set mapreduce.input.fileinputformat.split.minsize=2147483648;
set mapreduce.job.reduce.slowstart.completedmaps=1;
set hive.auto.convert.join=true;
set hive.groupby.skewindata=true;
set hive.exec.compress.output=true;
set mapred.output.compress=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.max.created.files=10000000;
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.parallel=true;
set hive.exec.parallel.thread.number=32;
"
setmulti_Dt="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
"

dt1=`date +"%Y-%m-%d" -d '-1 day'`
dt2=`date +"%Y-%m-%d" -d '-2 day'`
dt60=`date +"%Y-%m-%d" -d '-60 day'`
```



```shell
#!/bin/sh -e
#--------------------
# 极速版首页 用户共点击行为
#--------------------

source ./io.sh

table_fastjs_behavior="app.bh_uuid_to_click_jsapp"
table_cf_general="tmpr.behavior_click_uuid_cf_general_fastjs"
table_coview="tmpr.app_product_details_coview_fastjs"
path_cf_general="hdfs://ns1013/user/recsys/recpro/tmpr.db/behavior_click_uuid_cf_general_fastjs"
path_coview="hdfs://ns1013/user/recsys/recpro/tmpr.db/app_product_details_coview_fastjs"

echo "$dt1"
echo "$dt60"
echo "$table_fastjs_behavior"
echo "$table_cf_general"
echo "$table_coview"
echo "$path_cf_general"
echo "$path_coview"

sql_create="
create table if not exists $table_cf_general
(
sku string,
csku string,
weight bigint
)
partitioned by (dt string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
location '$path_cf_general';
"

echo "$sql_create"
hive -e "$sql_create"
if [ $? -ne 0 ];then
  echo "====== error create $table_cf_general ======";
  exit 1;
fi
echo "--- $table_cf_general dt=$dt1 created! ---"

sql_data="
insert overwrite table $table_cf_general partition (dt='$dt1')
SELECT sku, csku, weight
FROM(
		SELECT
			/*+ mapjoin(c)*/
			a.item_sku_id sku,
			b.item_sku_id csku,
			SUM(1) weight
		FROM 
    		(SELECT uuid, item_sku_id FROM $table_fastjs_behavior WHERE dt = '$dt1' and item_sku_id is not null)a
    	JOIN (SELECT uuid, item_sku_id FROM $table_fastjs_behavior WHERE dt = '$dt1' and item_sku_id is not null)b
    	ON (a.uuid = b.uuid)
		JOIN(
            SELECT uuid
            FROM(
            		SELECT uuid, SUM(1) AS rn
            		FROM(
            				SELECT  uuid, item_sku_id
            				FROM $table_fastjs_behavior
            				WHERE dt = '$dt1'
            				GROUP BY uuid, item_sku_id
            			)c1
            		GROUP BY uuid
            	)c2
            WHERE c2.rn <= 80
        )c
		ON (a.uuid = c.uuid)
		WHERE a.item_sku_id <> b.item_sku_id
		GROUP BY a.item_sku_id, b.item_sku_id
	)a
WHERE weight > 1
"

echo "$sql_data"
hive -e "$sql_data"
if [ $? -ne 0 ];then
  echo "====== error $table_cf_general ======";
  exit 1;
fi
echo "--- $table_cf_general dt=$dt1 over! ---"

sql_create2="
create table if not exists $table_coview
(
sku string,
csku string,
num bigint
)
location '$path_coview'
"

echo "$sql_create2"
hive -e "$sql_create2"
if [ $? -ne 0 ];then
  echo "====== error create $table_coview ======";
  exit 1;
fi
echo "--- $table_coview created! ---"

sql_data2="
insert overwrite table $table_coview
select * from
(
  select sku
        ,csku
        ,sum(weight) num
  from $table_cf_general
  where dt <= '$dt1'
  and dt >= '$dt60'
  group by sku,csku
)a
where num>2;
"

echo "$sql_data2"
hive -e "$sql_data2"
if [ $? -ne 0 ];then
  echo "====== error $table_coview ======";
  exit 1;
fi
echo "--- $table_coview over! ---"
```

共点击数据剪枝

```shell
#!/usr/bin/env bash


config="
set mapreduce.input.fileinputformat.split.maxsize=1547483648;
set mapreduce.input.fileinputformat.split.minsize=1547483648;
set mapreduce.job.reduces=200;
set mapreduce.job.reduce.slowstart.completedmaps=1;
set hive.auto.convert.join=true;
set hive.merge.mapfiles = true;
set hive.merge.mapredfiles = true;
set hive.merge.size.per.task = 256000000;
set hive.merge.smallfiles.avgsize = 104857600;
"

in_sku_information="recall.m_sku_information"

#project infomation
db="tmpr"
db_path="hdfs://ns1013/user/recsys/recpro/tmpr.db"
project="sk_gnn_relevance_click_part_40day_fastjs_dtWindow"

#output table
out_edge_with_cid3="${db}.${project}_edge_with_cid3"
out_edge_data="${db}.${project}_edge_data"
out_id_edge_data="${db}.${project}_id_edge_data"


# ============ 40天共点击数据

i=1
dt=`date +"%Y-%m-%d" -d "-$i day"`
dt40=`date +"%Y-%m-%d" -d "-$[${i}+40] day"`
out_app_product_details_coview=app.out_app_product_details_coview_40days_fastjs_dtWindow
today="'`date +"%Y-%m-%d" -d "-1 day"`'"

:<<!
!
HQL="
$config

create table if not exists ${out_app_product_details_coview}
(
sku string,
csku string,
num int
)
location '${db_path}/${out_app_product_details_coview}';

insert overwrite table ${out_app_product_details_coview}
select * from
(
select sku
,csku
,cast(sum(weight*1/(log(datediff(cast($today as string),dt)+2))) as int) as num
from tmpr.behavior_click_uuid_cf_general_fastjs
where dt<'$dt' and dt>='$dt40'
group by sku,csku
)a
where num>20;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 计算错误======";
    exit 1;
fi
echo "DICK"

#=================== 拼接cid3
HQL="
$config

create table if not exists ${out_edge_with_cid3}
(
    sku string,
    cid3 string,
    brand string,
    pw string,
    csku string,
    ccid3 string,
    cbrand string,
    cpw string,
    num bigint
)
row format delimited fields terminated by '\073'
location '${db_path}/${project}_edge_with_cid3';
"

echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 建表错误 ======";
  exit 1;
fi


#in_product_details_coview=app.app_product_details_coview
HQL="
$config
insert overwrite table ${out_edge_with_cid3}
select a.sku as sku, b.cid3 as cid3, b.brand as brand, b.pw as pw, a.csku as csku, c.cid3 as ccid3, c.brand as cbrand, c.pw as cpw, a.num as num from
        ${out_app_product_details_coview} a
    join
        (
            select distinct item_sku_id as sku, item_first_cate_cd as cid1, item_third_cate_cd as cid3,brand_code as brand, pwid_peanut as pw  from ${in_sku_information} where sku_valid_flag=1 and sku_status_cd='3001'
        ) b
    on a.sku=b.sku
    join
        (
            select distinct item_sku_id as sku,  item_first_cate_cd as ccid1, item_third_cate_cd as cid3,brand_code as brand,pwid_peanut as pw from ${in_sku_information} where sku_valid_flag=1 and sku_status_cd='3001'
        ) c
    on a.csku=c.sku
    where b.cid1 == c.ccid1 and b.sku is not null and c.sku is not null and b.cid3 not in ('16965','1364','9743','9744','14377','12008','9748','3493','1502','14699','1506','9313');
"

echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 拼接cid3错误 ======";
  exit 1;
fi


#################要求边的权重大于0.002
HQL="
$config

create table if not exists ${out_edge_data}
(
    sku string,
    cid3 string,
    brand string,
    pw string,
    csku string,
    ccid3 string,
    cbrand string,
    cpw string,
    coview float
)
row format delimited fields terminated by '\073'
location '${db_path}/${project}_edge_data';
"

echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 建表错误 ======";
  exit 1;
fi

HQL="
$config
insert overwrite table ${out_edge_data}
select c.sku as sku, c.cid3 as cid3,c.brand as brand, c.pw as pw, c.csku as csku, c.ccid3 as ccid3, c.cbrand as cbrand, c.cpw as cpw,c.coview as coview from
    (
        select a.sku as sku, a.cid3 as cid3, a.brand as brand, a.pw as pw,a.csku as csku, a.ccid3 as ccid3, a.cbrand as cbrand, a.cpw as cpw, a.num/b.sku_num as coview from
                ${out_edge_with_cid3} a
            join
                (
                    select sku, sum(num) as sku_num from ${out_edge_with_cid3} group by sku
                ) b
            on a.sku=b.sku
    ) c
    where c.coview>=0.001;
"

echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 筛选边数据错误 ======";
  exit 1;
fi
#exit


HQL="
$config

create table if not exists ${out_id_edge_data}
(
sku string,
cid3 string,
brand string,
pw string,
csku string,
ccid3 string,
cbrand string,
cpw string,
coview float
)

row format delimited fields terminated by '\073'
location '${db_path}/${project}_id_edge_data';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
echo "====== 建表错误 ======";
exit 1;
fi

HQL="
$config
insert overwrite table ${out_id_edge_data}
select sku,cid3,brand,pw,csku,ccid3,cbrand, cpw,coview from
(
select sku,cid3,brand,pw,csku,ccid3, cbrand,cpw, coview, row_number() over(partition by sku order by coview desc) as rank from ${out_edge_data}
) a
where rank<=95;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
echo "====== 筛选边数据错误 ======";
exit 1;
fi
```

共订单数据剪枝

```shell
#!/usr/bin/env bash
set -x

config="
set mapreduce.input.fileinputformat.split.maxsize=1547483648;
set mapreduce.input.fileinputformat.split.minsize=1547483648;
set mapreduce.job.reduce.slowstart.completedmaps=1;
set hive.auto.convert.join=true;
set hive.merge.mapfiles = true;
set hive.merge.mapredfiles = true;
set hive.merge.size.per.task = 256000000;
set hive.merge.smallfiles.avgsize = 144857600;
"

#input table
in_sku_information="recall.m_sku_information"
in_order="app.bh_uuid_to_ord_jsapp"

#project infomation
db="tmpr"
db_path="hdfs://ns1007/user/recsys/recpro/tmpr.db"
project="sk_gnn_relevance_order_dt180_no_c1_filter_fastjs"

#output table
out_past_order_data="${db}.${project}_past_order_data"
out_coorder_data="${db}.${project}_coorder_data"

out_point_data="${db}.${project}_point_data"
out_edge_data="${db}.${project}_edge_data"
out_id_edge_data="${db}.${project}_id_edge_data"



i=1
dt=`date +"%Y-%m-%d" -d "-$i day"`
dt180=`date +"%Y-%m-%d" -d "-$[${i}+180] day"`

:<<!
!
####################挑选出过去半年中的订单，并且过去三十天的订单数大于等于2的sku
HQL="
$config

create table if not exists ${out_past_order_data}
(
    uuid string,
    sku string,
    request_tm string,
    dt string
)
location '${db_path}/${project}_past_order_data';
"

echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 建表错误 ======";
  exit 1;
fi

HQL="
$config

insert overwrite table ${out_past_order_data}
select a.uuid as uuid, a.item_sku_id as sku, a.request_tm as rt, a.dt as dt from
        (
            select * from ${in_order} where dt>='${dt180}' and dt<='${dt}'
        ) a
    join
        (
            select item_sku_id as sku from ${in_sku_information} where cnt_order_30>=2 and sku_valid_flag=1 and sku_status_cd='3001'
        ) b
    on a.item_sku_id=b.sku
    join
        (
            select uuid, count(distinct item_sku_id) as order_num from ${in_order} where dt>='${dt180}' and dt<='${dt}' group by uuid
        ) c
    on a.uuid=c.uuid
    where c.order_num<=1000 and a.uuid is not null;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 筛选去过半年的购买订单错误 ======";
  exit 1;
fi

################生成共同购买的数据，至少有两次同购买

HQL="
$config
create table if not exists ${out_coorder_data}
(
    sku string,
    csku string,
    weight bigint
)
location '${db_path}/${project}_coorder_data';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 建表错误 ======";
  exit 1;
fi


HQL="
$config

insert overwrite table ${out_coorder_data}
select sku, csku, weight from (
select a.sku as sku,b.sku as csku, count(*) as weight from
${out_past_order_data} a
join
${out_past_order_data} b
on a.uuid = b.uuid
where a.sku <> b.sku and abs(datediff(a.dt,b.dt)) <=7 group by a.sku, b.sku
) res
where weight>1;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 生成同购买数据错误 ======";
  exit 1;
fi
#exit
##生成点数据，从coorder表中直接选出来sku，拼上特征
HQL="
$config

create table if not exists ${out_point_data}
(
    id int,
    sku string,
    cid1 string,
    cid2 string,
    cid3 string
)
row format delimited fields terminated by '\073'
location '${db_path}/${project}_point_data';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 建表错误 ======";
  exit 1;
fi

HQL="
$config

insert overwrite table ${out_point_data}
select id, sku, cid1, cid2, cid3  from (
    select row_number() over(partition by 1)-1 as id, a.sku as sku, b.cid1 as cid1, b.cid2 as cid2, b.cid3 as cid3
    from
        (
            select distinct sku from ${out_coorder_data}
        ) a
    join
        (
            select distinct item_sku_id, item_first_cate_cd as cid1, item_second_cate_cd as cid2, item_third_cate_cd as cid3, brand_code as brand_id, shop_id from ${in_sku_information}
        ) b
    on a.sku=b.item_sku_id
    where b.cid1 is not null
          and b.cid2 is not null
          and b.cid3 is not null) e
order by id;
"

echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 筛选点数据错误 ======";
  exit 1;
fi


##生成边数据
HQL="
$config

create table if not exists ${out_edge_data}
(
    src int,
    sku string,
    dst int,
    csku string,
    coview float
)
row format delimited fields terminated by '\073'
location '${db_path}/${project}_edge_data';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 建表错误 ======";
  exit 1;
fi

HQL="
$config
insert overwrite table ${out_edge_data}
select src, sku, dst, csku, coview from
    (
        select c.id as src, a.sku as sku, d.id as dst, a.csku as csku, a.weight/b.sku_num as coview from
            ${out_coorder_data} a
        join
            (
                select sku, sum(weight) as sku_num from ${out_coorder_data} group by sku
            ) b
        on a.sku=b.sku
        join
            (
                select id, sku from ${out_point_data}
            ) c
        on a.sku=c.sku
        join
            (
                select id, sku from ${out_point_data}
            ) d
        on a.csku=d.sku
    ) e
where e.coview>0.002;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 筛选边数据错误 ======";
  exit 1;
fi


#=============== 连接关系数据
HQL="
$config

create table if not exists ${out_id_edge_data}
(
    sku int,
    csku int,
    coview float
)

row format delimited fields terminated by '\073'
location '${db_path}/${project}_id_edge_data';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 建表错误 ======";
  exit 1;
fi

HQL="
$config
insert overwrite table ${out_id_edge_data}
select sku, csku, coview from
    (
        select sku, csku, coview, row_number() over(partition by sku order by coview desc) as rank from ${out_edge_data}
    ) a
where a.rank<=95;
"

echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
  echo "====== 筛选边数据错误 ======";
  exit 1;
fi
```

点击订单数据合并

```shell
db_path="hdfs://ns1013/user/recsys/recpro/tmpr.db"
project=click_order_dt180_40days_fastjs_dtWindow

click_table=tmpr.sk_gnn_relevance_click_part_40day_fastjs_dtWindow_id_edge_data
order_table=tmpr.sk_gnn_relevance_order_dt180_no_c1_filter_fastjs_id_edge_data

complement_table_same_c3_same_brand=tmpr.sk_gnn_sim_${project}_same_c3
complement_table_same_c3_diff_brand=tmpr.sk_gnn_sim_${project}_same_c3_diff_brand
complement_table_diff_c3=tmpr.sk_gnn_sim_${project}_diff_c3
union_table=tmpr.sk_gnn_sim_${project}
union_table_processed=tmpr.sk_gnn_sim_${project}_process_coview
union_id_sku=tmpr.sk_gnn_sim_${project}_id_sku
encoded_table=tmpr.sk_gnn_sim_${project}_encoded


config="
set mapreduce.input.fileinputformat.split.maxsize=1547483648;
set mapreduce.input.fileinputformat.split.minsize=1547483648;
set mapreduce.job.reduces=200;
set mapreduce.job.reduce.slowstart.completedmaps=1;
set hive.auto.convert.join=true;
set hive.merge.mapfiles = true;
set hive.merge.mapredfiles = true;
set hive.merge.size.per.task = 256000000;
set hive.merge.smallfiles.avgsize = 104857600;
"

:<<!
!
HQL="
$config

create table if not exists ${complement_table_same_c3_same_brand}
(
sku string,
csku string,
coview float
)

row format delimited fields terminated by '\073'
location '${db_path}/${complement_table_same_c3_same_brand}';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi

HQL="
insert overwrite table ${complement_table_same_c3_same_brand}
select sku, csku, coview*1.5 from ${click_table}
    where cid3 = ccid3 and cid3 not in ("16965","982","3493","1502","14699","1506","9313") and (brand=cbrand or pw =cpw);
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== DICK: complement breakdown ======";
    exit 1;
fi

#============ same c3 diff brand =================
HQL="
$config

create table if not exists ${complement_table_same_c3_diff_brand}
(
sku string,
csku string,
coview float
)

row format delimited fields terminated by '\073'
location '${db_path}/${complement_table_same_c3_diff_brand}';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi

HQL="
insert overwrite table ${complement_table_same_c3_diff_brand}
select sku, csku, coview from
(
select sku, csku, coview,row_number() over(partition by sku order by coview desc) as rank from ${click_table} where cid3 = ccid3 and cid3 not in ("16965","982","3493","1502","14699","1506","9313") and brand!=cbrand and pw !=cpw
)a
where rank <=7;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== DICK: complement breakdown ======";
    exit 1;
fi
#============ diff c3 =================
HQL="
$config

create table if not exists ${complement_table_diff_c3}
(
sku string,
csku string,
coview float
)

row format delimited fields terminated by '\073'
location '${db_path}/${complement_table_diff_c3}';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi

HQL="
insert overwrite table ${complement_table_diff_c3}
select sku, csku, coview from
(
select sku, csku, coview, row_number() over(partition by sku order by coview desc) as rank from ${click_table} where cid3 != ccid3 and cid3 not in ("16965","982","3493","1502","14699","1506","9313")
)a
where rank <=3;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
   echo "====== DICK: complement breakdown ======";
   exit 1;
fi


# i============================
HQL="
$config
create table if not exists ${union_table}
(
sku string,
csku string,
coview float
)

row format delimited fields terminated by '\073'
location '${db_path}/${union_table}';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi

HQL="
insert overwrite table ${union_table}
select sku, csku,coview from ${complement_table_same_c3_same_brand}
union
select sku, csku,coview from ${complement_table_same_c3_diff_brand}
union
select sku, csku,coview from ${complement_table_diff_c3}
union
select sku, csku,coview from ${order_table}
;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi
# ============= 合并 coview:
# ========= 考虑到clk和ord不平权，先分别计算coview，再简单加和
HQL="
$config
create table if not exists ${union_table_processed}
(
sku string,
csku string,
coview_processed float
)

row format delimited fields terminated by '\073'
location '${db_path}/${union_table_processed}';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi

HQL="
insert overwrite table ${union_table_processed}
select sku, csku, sum(coview) as coview_processed from ${union_table} group by sku,csku;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi
# =========================================================
#####建立sku字典
HQL="
$config
create table if not exists ${union_id_sku}
(
id_sku int,
sku string
)
location '${db_path}/${union_id_sku}';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi

HQL="
$config
insert overwrite table ${union_id_sku}
select row_number() over(partition by 1)-1 as id_sku, sku as sku from (
    select distinct sku from ${union_table} where sku != 'NULL'
) a;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建sku dict 错误 ======";
    exit 1;
fi

# =============================

HQL="
$config
create table if not exists ${encoded_table}
(
sku_id int,
csku_id int,
coview float
)
location '${db_path}/${encoded_table}';
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 建表错误 ======";
    exit 1;
fi

HQL="
$config

insert overwrite table ${encoded_table}
select sku,csku,coview from (
select b.id_sku as sku ,c.id_sku as csku, a.coview_processed as coview from ${union_table_processed} a
join
(select id_sku, sku from ${union_id_sku}) b
on a.sku = b.sku
join
(select id_sku, sku from ${union_id_sku}) c
on a.csku = c.sku
) h;
"
echo "${HQL}"
hive -e "${HQL}"
if [ $? -ne 0 ];then
    echo "====== 数据encode错误 ======";
    exit 1;
fi
```

Node2vec

游走方式可替换：https://github.com/shenweichen/GraphEmbedding/blob/master/ge/walker.py

```shell
set -x
dt=`date +"%Y%m%d%H%M"`
#exit
#out_directory="/user/recsys/recpro/songkai35/featureLog/i2i_features/sku_dim/dt=${dt}"
out_directory="hdfs://ns1013/user/recsys/recpro/rec_business_algo/project/fastjs/graph_recaller/node2vec/data/sk_gnn_sim_click_order_dt180_40days_fastjs_dtWindow"
hadoop fs -rm -r $out_directory

in_directory=hdfs://ns1013/user/recsys/recpro/tmpr.db/tmpr.sk_gnn_sim_click_order_dt180_40days_fastjs_dtWindow_encoded

mypython="hdfs://ns1013/user/recsys/recpro/rec_business_algo/tools/python27.tar.gz"
#-D mapred.job.priority=HIGH \
#    -D map.output.key.field.separator='\t' \
#    -D mapred.min.split.size=373741824 \
hadoop jar /software/servers/hadoop-2.7.1/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar \
    -archives "${mypython}#python" \
    -D mapred.job.map.capacity=1500 \
    -D mapred.job.reduce.capacity=2000 \
    -D mapred.map.tasks=2000 \
    -D mapred.reduce.tasks=2000 \
    -D stream.memory.limit=1024 \
    -D abaci.split.remote=true \
    -D mapred.map.over.capacity.allowed=true \
    -D mapred.job.priority='VERY_HIGH' \
    -D mapred.map.tasks.speculative.execution=true \
    -D mapred.job.map.memory.mb=2000 \
    -D mapred.job.reduce.memory.mb=20288 \
    -D mapred.task.timeout=600000000 \
    -input $in_directory \
    -output $out_directory \
    -mapper "python/python27/bin/python2 mapreduce.py map" \
    -reducer "python/python27/bin/python2 mapreduce.py red" \
    -file mapreduce.py

if [ $? -ne 0 ];then
     echo "M/R Job Info fails"
     exit 9
fi
#hadoop fs -touchz $out_directory/.done
echo "~~~sucessful~~~"
echo "Your outputDir is on HDFS:"${out_directory}
```

Node2vec

```python
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys
import networkx as nx
import numpy as np
import random


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		#print ('Walk iteration:')
		for walk_iter in range(num_walks):
			#print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]



def mapper():
    for line in sys.stdin:
        key = random.randint(0,100)
        #print str(key)+'\t'+line.rstrip('\n')
        parts = line.rstrip('\n').split('\001')
        #if parts[0]=="NULL" or parts[1]=="NULL": continue
        print str(key)+'\t'+ parts[0]+' ' + parts[1] +' ' + parts[2]

def reducer():
    # params of node2vec
    p = 1
    q = 1
    directed = False
    weighted = True
    num_walks = 3
    walk_length = 20
    input_ = []
    for line in sys.stdin:
        key,node = line.rstrip('\n').split('\t')
        #node = line.rstrip('\n')#.split('\t')
        input_.append(node)
    #print ("input_,",input_)
    if weighted:
        nx_G = nx.read_edgelist(input_, nodetype=str, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        nx_G = nx.read_edgelist(input_, nodetype=str, create_using=nx.DiGraph())
        for edge in nx_G.edges():
            nx_G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        nx_G = nx_G.to_undirected()

    G = Graph(nx_G,directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    for walk in walks:
        #print walk
        print ' '.join([str(e) for e in walk])


if __name__ == "__main__":
    if sys.argv[1] =='map':
        mapper()
    if sys.argv[1] == 'red':
        reducer()
```

Node2vec过滤

```shell
set -x
dt=`date +"%Y%m%d%H%M"`
#out_directory="hdfs://ns1013/user/recsys/recpro/songkai35/gnn/node2vec/data/output/homo.sim.v1.filter"
out_directory="hdfs://ns1013/user/recsys/recpro/songkai35/gnn/node2vec/data/output/homo.sim.mixed.3000w.dtWindow.superposition.brand.filter"
hadoop fs -rm -r $out_directory

in_directory="hdfs://ns1013/user/recsys/recpro/songkai35/gnn/node2vec/data/output/homo.sim.mixed.3000w.dtWindow.superposition.brand"

mypython="hdfs://ns1013/user/recsys/recpro/basic_alg/personal/houlinfang/tools/python27.tar.gz"
#-D mapred.job.priority=HIGH \
#    -D map.output.key.field.separator='\t' \
#    -D mapred.min.split.size=373741824 \
hadoop jar /software/servers/hadoop-2.7.1/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar \
    -archives "${mypython}#python" \
    -D mapred.job.map.capacity=1500 \
    -D mapred.job.reduce.capacity=2000 \
    -D mapred.map.tasks=1000 \
    -D mapred.reduce.tasks=30 \
    -D stream.memory.limit=1024 \
    -D abaci.split.remote=true \
    -D mapred.map.over.capacity.allowed=true \
    -D mapred.job.priority='VERY_HIGH' \
    -D mapred.map.tasks.speculative.execution=true \
    -D mapred.job.map.memory.mb=2000 \
    -D mapred.job.reduce.memory.mb=3096 \
    -D mapred.task.timeout=600000 \
    -input $in_directory \
    -output $out_directory \
    -mapper "python/python27/bin/python2 mapreduce_filter.py map" \
    -reducer "python/python27/bin/python2 mapreduce_filter.py red" \
    -file mapreduce_filter.py 

if [ $? -ne 0 ];then
     echo "M/R Job Info fails"
     exit 9
fi
#hadoop fs -touchz $out_directory/.done
echo "~~~sucessful~~~"
echo "Your outputDir is on HDFS:"${out_directory}
hadoop fs -rmr ${in_directory}
```

```python
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys
#import numpy as np


def mapper():
    window = 6
    seq_len =60
    for line in sys.stdin:
        #key = random.randint(0,999)
        try:
            words = line.rstrip().split(' ')
        except:
            continue
        #print words
        dct = {}
        for word in words:
            if dct.get(word) is None:
                dct.setdefault(word,1)
        size = len(dct)
        if size <= window:
            #print str(key)+'\t'+ parts[0]+' ' + parts[1] +' ' + parts[2]
            print ' '.join(words[:window+1])
        elif size > window and size <= 2*window:
            print ' '.join(words[:2*window+1])
        elif size >2*window and size <= 3*window:
            print ' '.join(words[:3*window+1])
        else:
            threshold = min(int(1.5*size),seq_len)
            print ' '.join(words[:threshold])


def reducer():
    # params of node2vec
    last_seq = ""
    for line in sys.stdin:
        seq =  line.rstrip('\n')
        if seq != last_seq:
            if last_seq != "":
                print last_seq
        last_seq = seq
    if last_seq:
        print last_seq


if __name__ == "__main__":
    if sys.argv[1] =='map':
        mapper()
    if sys.argv[1] == 'red':
        reducer()
```

补充内容

高点击数据生产

```shell
set -x
#exit
#out_directory="/user/recsys/recpro/songkai35/featureLog/i2i_features/sku_dim/dt=${dt}"
out_directory="hdfs://ns1013/user/recsys/recpro/songkai35/gnn/node2vec/data/high_clk/dt=40_days"
hadoop fs -rm -r $out_directory

#in_directory=hdfs://ns1013/user/recsys/recpro/tmpr.db/tmpr.sk_gnn_sim_click_order_dt7_50days_3000w_dtWindow_superposition_brand_encoded
#date_month_start=`date +"%Y-%m-01"`
#date_last_month_end=`date -d "${date_month_start} last day" +%Y-%m-%d`

today=`date +"%Y-%m-%d" -d "-1 day"`

dt=$(date -I -d "$today - 36 day")
paths="hdfs://ns1013/user/recsys/recpro/app.db/bh_uuid_to_click_app_valid/dt="${dt}

dt=$(date -I -d "$dt + 1 day")
while [ "$dt" != $today ]
do
  paths=${paths}",hdfs://ns1013/user/recsys/recpro/app.db/bh_uuid_to_click_app_valid/dt="${dt}
  dt=$(date -I -d "$dt + 1 day")
done
in_directory=$paths
mypython="hdfs://ns1013/user/recsys/recpro/houlinfang/tools/python27.tar.gz"
#-D mapred.job.priority=HIGH \
#    -D map.output.key.field.separator='\t' \
#    -D mapred.min.split.size=373741824 \
hadoop jar /software/servers/hadoop-2.7.1/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar \
    -archives "${mypython}#python" \
    -D mapred.job.map.capacity=1500 \
    -D mapred.job.reduce.capacity=2000 \
    -D mapred.map.tasks=1000 \
    -D mapred.reduce.tasks=1000 \
    -D stream.memory.limit=1024 \
    -D abaci.split.remote=true \
    -D mapred.map.over.capacity.allowed=true \
    -D mapred.job.priority='VERY_HIGH' \
    -D mapred.map.tasks.speculative.execution=true \
    -D mapred.job.map.memory.mb=2000 \
    -D mapred.job.reduce.memory.mb=25288 \
    -D mapred.task.timeout=600000000 \
    -input $in_directory \
    -output $out_directory \
    -mapper "python/python27/bin/python2 mapreduce_high_clk.py map" \
    -reducer "python/python27/bin/python2 mapreduce_high_clk.py red" \
    -file mapreduce_high_clk.py

if [ $? -ne 0 ];then
     echo "M/R Job Info fails"
     exit 9
fi
#hadoop fs -touchz $out_directory/.done
echo "~~~sucessful~~~"
echo "Your outputDir is on HDFS:"${out_directory}

```

```python
import sys

def mapper():
    for line in sys.stdin:
        parts = line.rstrip('\n').split('\001')
        uuid = parts[0]
        item_sku_id = parts[1]
        timeStamp = parts[2]

        print '\t'.join([uuid,item_sku_id+","+timeStamp])

def reducer():
    last_key = ''
    tmp = ''
    cnt = 0
    threshold = 5
    for line in sys.stdin:
        key, val= line.rstrip().split('\t')
        if key != last_key:
            if last_key and cnt >threshold:
                print last_key + '\t' + tmp.rstrip(' ')#+" \t"+str(cnt)
            tmp = val +" "
            cnt =1
            last_key = key
        else:
            tmp += val+" "
            cnt +=1 
    if last_key and cnt >threshold:
        print last_key + '\t' + tmp.rstrip(' ')

if __name__ == "__main__":
    if sys.argv[1] =='map':
        mapper()
    if sys.argv[1] == 'red':
        reducer()
```

高点击序列生产

```shell
set -x
dt=`date +"%Y%m%d%H%M"`
#exit
#out_directory="/user/recsys/recpro/songkai35/featureLog/i2i_features/sku_dim/dt=${dt}"
out_directory="hdfs://ns1013/user/recsys/recpro/songkai35/gnn/node2vec/data/high_clk.processed_series/dt=40_days"
hadoop fs -rm -r $out_directory

#in_directory=hdfs://ns1013/user/recsys/recpro/tmpr.db/tmpr.sk_gnn_sim_click_order_dt7_50days_3000w_dtWindow_superposition_brand_encoded
in_directory="hdfs://ns1013/user/recsys/recpro/songkai35/gnn/node2vec/data/high_clk/dt=40_days"

mypython="hdfs://ns1013/user/recsys/recpro/songkai35/tools/python27.tar.gz"
#-D mapred.job.priority=HIGH \
#    -D map.output.key.field.separator='\t' \
#    -D mapred.min.split.size=373741824 \
hadoop jar /software/servers/hadoop-2.7.1/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar \
    -archives "${mypython}#python" \
    -D mapred.job.map.capacity=1500 \
    -D mapred.job.reduce.capacity=2000 \
    -D mapred.map.tasks=1000 \
    -D mapred.reduce.tasks=100 \
    -D stream.memory.limit=1024 \
    -D abaci.split.remote=true \
    -D mapred.map.over.capacity.allowed=true \
    -D mapred.job.priority='VERY_HIGH' \
    -D mapred.map.tasks.speculative.execution=true \
    -D mapred.job.map.memory.mb=2000 \
    -D mapred.job.reduce.memory.mb=25288 \
    -D mapred.task.timeout=600000000 \
    -input $in_directory \
    -output $out_directory \
    -mapper "python/python27/bin/python2 generate_series.py map" \
    -reducer "python/python27/bin/python2 generate_series.py red" \
    -file generate_series.py

if [ $? -ne 0 ];then
     echo "M/R Job Info fails"
     exit 9
fi
#hadoop fs -touchz $out_directory/.done
echo "~~~sucessful~~~"
echo "Your outputDir is on HDFS:"${out_directory}
```

```python
import sys
from datetime import datetime

def mapper():
    threshold = 5
    for line in sys.stdin:
        parts = line.rstrip('\n').split('\t')
        uuid = parts[0]
        sku_timeStamps = parts[1].split(' ')
        #print sku
        last_sku = ""
        skus = ""
        last_dateTime = ""
        for sku_timeStamp in sorted(sku_timeStamps,key = lambda y:int(y.split(',')[-1])):
            sku = sku_timeStamp.split(',')[0]
            timeStamp =  int(sku_timeStamp.split(',')[-1])/1000
            #dateTime = datetime.utcfromtimestamp(timeStamp).strftime('%Y-%m-%d %H:%M:%S')
            dateTime = datetime.utcfromtimestamp(timeStamp).strftime('%Y-%m-%d')
            if dateTime != last_dateTime:
                if last_dateTime and len(skus.split(' '))>threshold:
                   print skus.rstrip(' ')
                skus = sku+" "
                last_dateTime = dateTime
                last_sku = sku
            else:
                if sku != last_sku:
                    skus += sku+" "
                    last_sku = sku
        if last_dateTime and len(skus.split(' '))>threshold:
            print (skus.rstrip(' '))
        #print '\t'.join([uuid,item_sku_id+","+timeStamp])

def reducer():
    for line in sys.stdin:
        print (line.rstrip())
if __name__ == "__main__":
    if sys.argv[1] =='map':
        mapper()
    if sys.argv[1] == 'red':
        reducer()

```

w2v模型训练





倒排处理









