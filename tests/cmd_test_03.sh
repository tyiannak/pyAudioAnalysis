data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py trainHMMsegmenter_fromdir -i \"${data_root}/radioFinal/train/\" -o tempHMM -mw 1.0 -ms 1.0"
echo $command1
eval $command1
command2="python3 ../pyAudioAnalysis/audioAnalysis.py segmentClassifyFileHMM -i \"${data_root}/pyAudioAnalysis/data/scottish.wav\" --hmm tempHMM"
echo $command2
eval $command2
