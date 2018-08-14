data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py segmentationEvaluation --model svm --modelName \"${data_root}//pyAudioAnalysis/data/svmSM\"  -i \"${data_root}/radioFinal/test\""
echo $command1
eval $command1
command2="python3 ../pyAudioAnalysis/audioAnalysis.py segmentationEvaluation --model knn --modelName \"${data_root}//pyAudioAnalysis/data/knnSM\"  -i \"${data_root}/radioFinal/test\""
echo $command2
eval $command2
command3="python3 ../pyAudioAnalysis/audioAnalysis.py segmentationEvaluation --model hmm --modelName \"${data_root}//pyAudioAnalysis/data/hmmRadioSM\"  -i \"${data_root}/radioFinal/test\""
echo $command3
eval $command3
