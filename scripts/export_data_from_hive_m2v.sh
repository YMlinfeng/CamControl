#!/bin/bash
if [ $# -ne 2 -a $# -ne 3 ]; then
	echo "please input the sql you want to export and the destination file"
	exit
fi
sql="$1"
export_file=$2

enable_json=false
if [ $# -eq 3 -a "$3" = "enable_json" ]; then
	enable_json=true
fi

if [ ! -f $export_file ]; then
	dir_of_export_file=$(echo $(dirname $export_file))
	if [ ! -d $dir_of_export_file ]; then
		mkdir -p $dir_of_export_file
	fi
fi

if [ $enable_json == true ]; then
	hive -e "$sql" >$export_file
	#过滤掉hive的命令信息输出
	sed -i '/^[0-9]\{4\}-[0-9]\{1,2\}-[0-9]\{1,2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\} \(信息\|警告\):\| \(A\|P\)M \(INFO\|WARNING\):\|TaskAttemptContextImpl/d' $export_file
	exit
fi

export_folder="/home/mmu/data/hive_export_data_folder_"$(date +%s)
mkdir $export_folder

echo "export sql is:"$sql

echo "destination file  is:"$export_file
echo "begin to export"

begin=$(date +%s)
/home/hadoop/software/hive/bin/beeline -u 'jdbc:hive2://hiveserver-zk1.internal:2181,hiveserver-zk2.internal:2181,hiveserver-zk3.internal:2181,hiveserver-zk4.internal:2181,hiveserver-zk5.internal:2181/default;serviceDiscoveryMode=zooKeeper;zooKeeperNamespace=hiveserver2?kuaishou.bigdata.job.authc.principal=mmu_vcg/project@kuaishou.com;kuaishou.bigdata.job.realuser=liqi12;task.group.id=1916;kuaishou.bigdata.job.authz.type=PROJECT;kuaishou.bigdata.job.authc.token=ChxtbXVfdmNnL3Byb2plY3RAa3VhaXNob3UuY29tGg0xMC4yOC4yMzIuMTc4IgQyNTkzKIqYxOL7MDCK6NCYwExAAkpUeyJzZXJ2aWNlU3RhZ2UiOiJQUk9EIiwiaXAiOiIxMC4yOC4yMzIuMTc4Iiwic2VydmljZUNhdGFsb2ciOiJtbXUuaW50ZWdyYXRlLnJ1bm5lciJ9.1S-j6rm3h0FBBxfHyMZp2pvh5zEEzB9eFBXwTevjlHU' -n mmu_vcg --outputformat=tsv2 --showHeader=false -e "$sql;" >$export_file
#/home/hadoop/software/apache-hive-2.3.2U2-bin/bin/hive -e "set hive.mapred.mode= unstrict;set parquet.column.index.access=true; insert overwrite local directory '$export_folder'\
#        row format delimited\
#        fields terminated by '\t'\
#        $sql "
#hive -e "$sql" >> $export_file
end=$(date +%s)
#echo "cost $[end-begin] s"

#cat $(echo $export_folder"/*") > $export_file
#rm -rf $export_folder

echo "export_finish"
