line=`ps aux|grep 43519|grep -v "grep"|wc -l`
while [ $line -eq 1 ]
do
    echo "no~"
    sleep 30
    line=`ps aux|grep 43519|grep -v "grep"|wc -l`
done
