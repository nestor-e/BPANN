import re
import sys

def main(fileName):
    f = open(fileName, "r", encoding="utf-8")
    next(f)
    for line in f:
        m = re.findall("[a-z]+=([a-z])", line)
        a = [0] * 26;
        for i in range(len(m)):
            idx = (ord(m[i]) - ord("a"))%26
            a[idx] = i + 1
        print("{", a[0], end="", sep="")
        for i in range(1, len(a)):
            print(",",a[i], end="", sep="")
        print("},")

if __name__ == "__main__":
    main(sys.argv[1])
