#include <vector>

typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;


class Layer{
	private:
		Layer* next;
		Layer* prev;
		int outputSize;
		int inputSize;
		bool useBias;
		Matrix weights;
		Vector bias;
		Vector beta;
		Vector out;
		Vector delta;
		void forward();
		void backPropagateDelta();
	public:
		Layer(int size, Layer prev, bool useBias);
		Layer(int outSize, int inSize, bool useBias);
        double sigmoid(double t, int i);
        void randomInit();
		void forward(Vector input);
		void backPropagateDelta(Vector expected);
		Vector getResults();
		void updateWeights(double speed);
}
