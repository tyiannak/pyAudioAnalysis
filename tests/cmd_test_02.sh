data_root=$1
echo $data_root
command="python3 ../pyAudioAnalysis/audioAnalysis.py trainClassifier -i \"${data_root}/SM/speech\" \"${data_root}/SM/music\"  --method svm_rbf -o svmSM"
echo $command
eval $command

command="python3 ../pyAudioAnalysis/audioAnalysis.py trainClassifier -i \"${data_root}/SM/speech\" \"${data_root}/SM/music\"  --method knn -o knnSM"
echo $command
eval $command