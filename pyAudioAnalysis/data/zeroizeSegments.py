import csv
import sys

file = open(sys.argv[1], 'rb')
csvreader = csv.reader(file)

firstline = csvreader.next()
subtractionVal = float(firstline[0])
# print 'SUB: ', subtractionVal
newStart = 0.0
end = float(firstline[1])
newEnd = end - subtractionVal
print str(newStart) + ',' + str(newEnd) + ',' + firstline[2]

while(True):
  try:
    row = csvreader.next()
    start = float(row[0])
    end = float(row[1])
    speaker = row[2]
    newStart = start - subtractionVal
    newEnd = end - subtractionVal
    print str(newStart) + ',' + str(newEnd) + ',' + speaker
  except:
    break
