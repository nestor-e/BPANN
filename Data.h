#import <vector>
#import <tuple>

typedef std::tuple<std::vector<double> , std::vector<double>> Datum;

class Data{
    private:
        int type; // 0 = minst, 1 = Mushrooms
        bool ready;
        std::vector< Datum > trainingValues;
        std::vector< Datum > testValues;
        bool initMINST(char* fileName);
        bool initMush(char* fileName);
    public:
        Data(int type);
        bool init(char* fileName);
        Datum testItem(int idx);
        Datum trainItem(int idx);
        int testCount();
        int trainCount();
        int getInputSize();
        int getOutoutSize();
}
