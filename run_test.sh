
python3 synthesize.py --text '大数据、云计算、物联网、人工智能等新一代信息技术的应用，给我们带来便利的同时，也带来了新的网络威胁。' --speaker_id 162 --restore_step $1 --mode single  -p $2/preprocess.yaml -m $2/model.yaml -t $2/train.yaml
python3 synthesize.py --text '去年开始，工信部、公安部、中央网信办三个中央主管部门多次通报或下架因为过度收集个人隐私的应用程序.' --speaker_id 162 --restore_step $1 --mode single  -p $2/preprocess.yaml -m $2/model.yaml -t $2/train.yaml
python3 synthesize.py --text '我们可以看到中央政府部门对应用程序的监管越来越严格.' --speaker_id 162 --restore_step $1 --mode single  -p $2/preprocess.yaml -m $2/model.yaml -t $2/train.yaml
