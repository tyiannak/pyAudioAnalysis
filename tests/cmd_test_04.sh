data_root=$1
echo $data_root
command1="python3 ../pyAudioAnalysis/audioAnalysis.py trainHMMsegmenter_fromfile -i \"${data_root}//pyAudioAnalysis/data/count.wav\"  --ground \"${data_root}//pyAudioAnalysis/data/count.segments\"  -o hmmcount -mw 0.1 -ms 0.1"
echo $command1
eval $command1
command2="python3 ../pyAudioAnalysis/audioAnalysis.py segmentClassifyFileHMM -i \"${data_root}/pyAudioAnalysis/data/count2.wav\" --hmm hmmcount"
echo $command2
eval $command2
