package plotter

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

type IPlotter[X float64, T plot.Plot] interface {
	PlotScatter(xData, yData []X, xLabel, yLabel string) *T
	PlotHistogram(data []X, label string) *T
	PairPlot(data [][]X, labels []string)
	PlotModel(xData []float64, yData []float64, weights []float64, bias float64, feature, label string)
	PlotMetrics(numEpochs int, metricsMap map[string][]float64)
}

type FloatPlotter struct{}

func NewPlotter() IPlotter[float64, plot.Plot] {
	return &FloatPlotter{}
}

func (p *FloatPlotter) PlotScatter(xData, yData []float64, xLabel, yLabel string) *plot.Plot {
	pts := make(plotter.XYs, len(xData))
	for i := range xData {
		if math.IsNaN(xData[i]) {
			pts[i].X = 0
		} else {
			pts[i].X = xData[i]
		}
		if math.IsNaN(yData[i]) {
			pts[i].Y = 0
		} else {
			pts[i].Y = yData[i]
		}
	}

	plot := plot.New()
	plot.Title.Text = yLabel + " vs " + xLabel
	plot.X.Label.Text = xLabel
	plot.Y.Label.Text = yLabel

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		log.Panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Points(2)
	scatter.GlyphStyle.Color = color.RGBA{B: 255, A: 255}
	plot.Add(scatter)

	return plot
}

func (p *FloatPlotter) PlotHistogram(data []float64, label string) *plot.Plot {
	plot := plot.New()
	plot.Title.Text = "Distribution of " + label
	plot.X.Label.Text = label

	for i := range data {
		if math.IsNaN(data[i]) {
			data[i] = 0
		}
	}

	hist, err := plotter.NewHist(plotter.Values(data), 16)
	if err != nil {
		log.Panic(err)
	}
	hist.FillColor = color.RGBA{R: 100, G: 150, B: 200, A: 255}
	plot.Add(hist)

	return plot
}

func (p *FloatPlotter) PairPlot(data [][]float64, labels []string) {
	gridSize := len(labels)
	tileSize := 4 * vg.Inch
	imgWidth := tileSize * vg.Length(gridSize)
	imgHeight := tileSize * vg.Length(gridSize)
	img := vgimg.New(imgWidth, imgHeight)
	dc := draw.New(img)

	for row, yCol := range data {
		for col, xCol := range data {

			var plot *plot.Plot
			if row == col {
				data := make([]float64, len(xCol))
				copy(data, xCol)
				plot = p.PlotHistogram(data, labels[col])
			} else {
				xData := make([]float64, len(xCol))
				copy(xData, xCol)
				yData := make([]float64, len(yCol))
				copy(yData, yCol)
				plot = p.PlotScatter(xData, yData, labels[col], labels[row])
			}

			// Posizioniamo il grafico nella griglia
			xStart := vg.Length(col) * tileSize
			yStart := vg.Length(gridSize-row-1) * tileSize
			canvas := draw.Canvas{
				Canvas: dc,
				Rectangle: vg.Rectangle{
					Min: vg.Point{X: xStart, Y: yStart},
					Max: vg.Point{X: xStart + tileSize, Y: yStart + tileSize},
				},
			}
			plot.Draw(canvas)
		}
	}

	err := os.Mkdir("plot", os.ModePerm)
	if err != nil {
		fmt.Printf("error creating directory: %v", err)
	}

	w, err := os.Create(fmt.Sprintf("./plot/data_%s.png", strings.Join(labels, "_")))
	if err != nil {
		log.Panic(err)
	}
	defer w.Close()

	_, err = vgimg.PngCanvas{Canvas: img}.WriteTo(w)

	if err != nil {
		log.Panic(err)
	}
}

// Plots the data on a cartesian plane with the feature on the x-axis and the label on the y-axis
func (p *FloatPlotter) PlotData(xData []float64, yData []float64, feature, label string) {
	points := make(plotter.XYs, len(xData))
	for i := range xData {
		points[i].X = xData[i]
		points[i].Y = yData[i]
	}

	plot := plot.New()
	plot.Title.Text = "Data Plot"
	plot.X.Label.Text = feature
	plot.Y.Label.Text = label

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		log.Fatalf("Failed to create scatter plot: %v", err)
	}
	scatter.GlyphStyle.Radius = vg.Points(3)

	plot.Add(scatter)
	if err := plot.Save(8*vg.Inch, 4*vg.Inch, fmt.Sprintf("./plot/data_%s_%s.png", feature, label)); err != nil {
		log.Fatalf("Failed to save plot: %v", err)
	}
}

func (p *FloatPlotter) PlotModel(xData []float64, yData []float64, weights []float64, bias float64, feature, label string) {
	if len(xData) != len(yData) {
		log.Fatalf("Mismatch between feature and label lengths")
	}

	// Create the line for the model
	linePoints := make(plotter.XYs, len(xData))

	points := make(plotter.XYs, len(xData))
	for i := range xData {
		points[i].X = xData[i]
		points[i].Y = yData[i]

		linePoints[i].X = xData[i]
		linePoints[i].Y = weights[0]*xData[i] + bias
	}

	plot := plot.New()
	plot.Title.Text = "Model Plot"
	plot.X.Label.Text = feature
	plot.Y.Label.Text = label

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		log.Fatalf("Failed to create scatter plot: %v", err)
	}
	scatter.GlyphStyle.Radius = vg.Points(3)

	line, err := plotter.NewLine(linePoints)
	if err != nil {
		log.Fatalf("Failed to create line plot: %v", err)
	}
	line.LineStyle.Width = vg.Points(2)
	line.LineStyle.Color = plotutil.Color(1)

	plot.Add(scatter, line)

	err = os.Mkdir("plot", os.ModePerm)
	if err != nil {
		fmt.Printf("error creating directory: %v", err)
	}

	if err := plot.Save(8*vg.Inch, 4*vg.Inch, "./plot/model_plot.png"); err != nil {
		log.Fatalf("Failed to save plot: %v", err)
	}
}

func (p *FloatPlotter) PlotMetrics(numEpochs int, metricsMap map[string][]float64) {
	epochs := make([]float64, numEpochs)
	for i := 0; i < numEpochs; i++ {
		epochs[i] = float64(i + 1)
	}

	pl := plot.New()

	pl.Title.Text = "Final Moving Average Loss Across Epochs"
	pl.X.Label.Text = "Epochs"
	pl.Y.Label.Text = "Loss"

	addLine := func(p *plot.Plot, name string, xData, yData []float64, lineColor color.Color) {
		points := make(plotter.XYs, len(xData))
		for i := range points {
			points[i].X = xData[i]
			points[i].Y = yData[i]
		}

		line, err := plotter.NewLine(points)
		if err != nil {
			log.Fatalf("Error creating line plot: %v", err)
		}
		line.LineStyle.Width = vg.Points(2)
		line.LineStyle.Color = lineColor

		p.Add(line)
		p.Legend.Add(name, line)
	}

	colorMap := map[int]color.Color{
		0: color.RGBA{R: 255, A: 255},
		1: color.RGBA{G: 255, A: 255},
		2: color.RGBA{B: 255, A: 255},
	}
	i := 0
	for metricName, metricValues := range metricsMap {
		addLine(pl, metricName, epochs, metricValues, colorMap[i])
		i++
	}

	os.Mkdir("plot", os.ModePerm)

	if err := pl.Save(10*vg.Inch, 5*vg.Inch, "./plot/comparison_metrics.png"); err != nil {
		log.Fatalf("Error saving plot: %v", err)
	}

	log.Println("Plot saved as comparison_metrics.png")

}
