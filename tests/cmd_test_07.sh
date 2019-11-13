data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py silenceRemoval -i \"${data_root}//pyAudioAnalysis/data/recording3.wav\"   --smoothing 1.0 --weight 0.3"
echo $command1
eval $command1
command2="python3 ../pyAudioAnalysis/audioAnalysis.py classifyFolder -i \"${data_root}/pyAudioAnalysis/data/recording3_\" --model svm --classifier \"${data_root}/pyAudioAnalysis/data/svmSM\" --detail"
echo $command2
eval $command2

