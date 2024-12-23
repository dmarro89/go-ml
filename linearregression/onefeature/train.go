package linear

import (
	"flag"
	"fmt"
	"strconv"

	"github.com/dmarro89/go-ml/dataset"
	"github.com/dmarro89/go-ml/plotter"
	"github.com/go-gota/gota/dataframe"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/janpfeifer/must"
)

var (
	flagCSVFile      = flag.String("csv_file", "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv", "Path to the CSV file containing training data")
	flagNumEpochs    = flag.Int("epochs", 20, "Number of epochs")
	flagLearningRate = flag.Float64("learning_rate", 0.001, "Initial learning rate.")
	flagBatchSize    = flag.Int("batch_size", 50, "Batch size for training")
)

const FARE_COLUMN = "FARE"
const TRIP_MILES_COLUMN = "TRIP_MILES"

func modelGraph(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
	_ = spec
	logits := layers.Dense(ctx, inputs[0], true, 1)
	return []*graph.Node{logits}
}

func OneFeature() {
	flag.Parse()

	// Load the dataset and print the summary statistics
	dataSet := dataset.NewDataFrameDataset()
	df := dataSet.GetDataFromURL(*flagCSVFile)
	columns := []string{"TRIP_MILES", "TRIP_SECONDS", "FARE", "TIP_RATE"}
	df = dataSet.ExtractColumns(df, columns)
	dataSet.DescribeData(df)

	// Get and print the correlation matrix
	corrMat := dataSet.GetCorrelationMatrix(df)
	corrMat.Print(columns)

	// Create a pair plot
	plot := plotter.NewPlotter()
	plot.PairPlot(dataSet.ToFloatMatrix(df), columns)

	// Create tensors from the dataframe just for the columns TRIP_MILES and FARE
	inputs, labels := dataframeToSingleTensors(df)
	fmt.Printf("Training data (inputs, labels): (%s, %s)\n\n", inputs.Shape(), labels.Shape())

	// Create a new backend
	backend := backends.New()
	// Create an in-memory dataset from the tensors.
	dataset, err := data.InMemoryFromData(backend, "Dataset", []any{inputs}, []any{labels})
	if err != nil {
		panic(fmt.Errorf("error creating dataset: %v", err))
	}

	dataset = dataset.Shuffle().BatchSize(*flagBatchSize, false)
	// Create context with learning rate
	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate)

	// Create a new trainer
	trainer := train.NewTrainer(backend, ctx, modelGraph,
		losses.MeanSquaredError,
		optimizers.StochasticGradientDescent(),
		nil, nil)

	loop := train.NewLoop(trainer)
	attachEpochMetricsCollector(loop)
	commandline.AttachProgressBar(loop)

	// Train the model
	_, err = loop.RunEpochs(dataset, *flagNumEpochs)
	if err != nil {
		panic(fmt.Errorf("error training model: %v", err))
	}

	// Get the learned coefficients and bias
	coefVar, biasVar := ctx.GetVariableByScopeAndName("/dense", "weights"), ctx.GetVariableByScopeAndName("/dense", "biases")
	learnedCoef, learnedBias := coefVar.Value(), biasVar.Value()
	fmt.Printf("Learned coefficients: %0.5f\n", learnedCoef.Value())
	fmt.Printf("Learned bias: %0.5f\n", learnedBias.Value())

	// Print model and metrics
	plot.PlotModel(df.Col(TRIP_MILES_COLUMN).Float(), df.Col(FARE_COLUMN).Float(), learnedCoef.Value().([][]float64)[0], learnedBias.Value().([]float64)[0], TRIP_MILES_COLUMN, FARE_COLUMN)
	plot.PlotMetrics(*flagNumEpochs, metricsMap)
}

func dataframeToSingleTensors(df dataframe.DataFrame) (inputs, labels *tensors.Tensor) {
	numRows := df.Nrow()

	inputData := make([]float64, numRows)
	labelData := make([]float64, numRows)

	for i := 0; i < numRows; i++ {
		row := df.Subset(i)
		labelValue := row.Col(FARE_COLUMN).Float()[0]
		labelData[i] = labelValue

		featureIdx := 0
		value := row.Col(TRIP_MILES_COLUMN).Float()[0]
		inputData[i+featureIdx] = value
		featureIdx++
	}

	inputs = tensors.FromFlatDataAndDimensions(inputData, numRows, 1)
	labels = tensors.FromFlatDataAndDimensions(labelData, numRows, 1)
	return inputs, labels
}

var metricsMap = map[string][]float64{}

func attachEpochMetricsCollector(loop *train.Loop) {
	currentEpoch := 0

	loop.OnStep("EpochMetrics", 100, func(loop *train.Loop, metrics []*tensors.Tensor) error {
		if loop.LoopStep > 0 && loop.LoopStep%*flagBatchSize == 0 {
			currentEpoch++
			fmt.Printf("Epoch %d completed. Metrics:\n", currentEpoch)

			trainMetrics := loop.Trainer.TrainMetrics()
			for i, metric := range metrics {
				metricsMap[trainMetrics[i].Name()] = append(metricsMap[trainMetrics[i].Name()], must.M1(strconv.ParseFloat(trainMetrics[i].PrettyPrint(metric), 64)))
			}
		}
		return nil
	})
}
