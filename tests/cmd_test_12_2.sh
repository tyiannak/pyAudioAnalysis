data_root=$1
echo $data_root

command1="python3 script_test_classifier.py -d \"${data_root}/\"  -c svm"; echo $command1; eval $command1

command2="python3 ../pyAudioAnalysis/audioAnalysis.py classifyFile -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\"  --model svm --classifier sm_svm"
echo $command2; eval $command2
command3="python3 ../pyAudioAnalysis/audioAnalysis.py segmentClassifyFile -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\"  --model svm --modelName sm_svm"
echo $command3; eval $command3
