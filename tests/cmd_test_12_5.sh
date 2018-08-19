data_root=$1
echo $data_root

command1="python3 script_test_classifier.py -d \"${data_root}/\"  -c randomforest"; echo $command1; eval $command1

command2="python3 ../pyAudioAnalysis/audioAnalysis.py classifyFile -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\"  --model randomforest --classifier sm_randomforest"
echo $command2; eval $command2
command3="python3 ../pyAudioAnalysis/audioAnalysis.py segmentClassifyFile -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\"  --model randomforest --modelName sm_randomforest"
echo $command3; eval $command3