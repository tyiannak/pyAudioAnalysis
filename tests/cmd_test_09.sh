data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py trainRegression -i \"${data_root}//pyAudioAnalysis/data/speechEmotion/\"  --method svm -o \"${data_root}//pyAudioAnalysis/data/svmSpeechEmotion\""
echo $command1
eval $command1