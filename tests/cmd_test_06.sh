data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py segmentationEvaluation --model svm --modelName \"${data_root}//models/svm_rbf_sm\"  -i \"${data_root}/radioFinal/test\""
echo $command1
eval $command1
command2="python3 ../pyAudioAnalysis/audioAnalysis.py segmentationEvaluation --model knn --modelName \"${data_root}//models/knn_sm\"  -i \"${data_root}/radioFinal/test\""
echo $command2
eval $command2
command3="python3 ../pyAudioAnalysis/audioAnalysis.py segmentationEvaluation --model hmm --modelName \"${data_root}//hmmRadioSM\"  -i \"${data_root}/radioFinal/test\""
echo $command3
eval $command3
