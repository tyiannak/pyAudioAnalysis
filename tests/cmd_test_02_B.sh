data_root=$1
echo $data_root
command="python3 ../pyAudioAnalysis/audioAnalysis.py trainClassifier -i \"${data_root}/musicalGenreClassification/Blues\" \"${data_root}/musicalGenreClassification/Classical\" \"${data_root}/musicalGenreClassification/Electronic\" \"${data_root}/musicalGenreClassification/Jazz\" \"${data_root}/musicalGenreClassification/Rap\" \"${data_root}/musicalGenreClassification/Rock\"  --method svm_rbf -o svmMusicGenre6  --beat"
echo $command
eval $command

command="python3 ../pyAudioAnalysis/audioAnalysis.py trainClassifier -i \"${data_root}/musicalGenreClassification/Blues\" \"${data_root}/musicalGenreClassification/Classical\" \"${data_root}/musicalGenreClassification/Electronic\" \"${data_root}/musicalGenreClassification/Jazz\" \"${data_root}/musicalGenreClassification/Rap\" \"${data_root}/musicalGenreClassification/Rock\"  --method knn -o knnMusicGenre6  --beat"
echo $command
eval $command