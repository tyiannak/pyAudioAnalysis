function matSegToCSV(matFileName, csvFileName)

load(matFileName)
fp = fopen(csvFileName, 'w');
for i=1:size(segs_r)    
    if classes_r(i) == 'M' className = 'music'; end
    if classes_r(i) == 'E' className = 'speech'; end
    if classes_r(i) == 'S' className = 'speech'; end
    fprintf(fp,'%.2f,%.2f,%s\n',segs_r(i,1), segs_r(i,2), className);
end
fclose(fp);