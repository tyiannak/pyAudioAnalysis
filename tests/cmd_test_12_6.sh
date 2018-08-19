data_root=$1
echo $data_root

command1="python3 script_test_classifier.py -d \"${data_root}/\"  -c extratrees"; echo $command1; eval $command1

command2="python3 ../pyAudioAnalysis/audioAnalysis.py classifyFile -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\"  --model extratrees --classifier sm_extratrees"
echo $command2; eval $command2
command3="python3 ../pyAudioAnalysis/audioAnalysis.py segmentClassifyFile -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\"  --model extratrees --modelName sm_extratrees"
echo $command3; eval $command3
