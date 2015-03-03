function matSegToCSV_dir(dirName)

d = dir([dirName filesep '*.mat']);

for i=1:length(d)
    curName = [dirName filesep d(i).name]
    matSegToCSV(curName, strrep(curName,'_true.mat', '.segments'));
end