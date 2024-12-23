package dataset

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/olekukonko/tablewriter"
	"gonum.org/v1/gonum/stat"
)

type CorrelationMatrix [][]float64

type IDataset[T dataframe.DataFrame] interface {
	GetDataFromURL(url string) T
	DescribeData(df T)
	ExtractColumns(df T, columnNames []string) T
	GetCorrelationMatrix(df T) CorrelationMatrix
	ToFloatMatrix(df T) [][]float64
}

type DataFrameDataset struct{}

func NewDataFrameDataset() IDataset[dataframe.DataFrame] {
	return &DataFrameDataset{}
}

func (d *DataFrameDataset) GetDataFromURL(url string) dataframe.DataFrame {
	response, err := http.Get(url)
	if err != nil {
		log.Fatal(err)
	}
	defer response.Body.Close()
	return dataframe.ReadCSV(response.Body)
}

func (d *DataFrameDataset) DescribeData(df dataframe.DataFrame) {
	// Print the dimensions of the dataframe
	fmt.Println(df.Dims())

	// Print the names of the columns
	fmt.Println(df.Names())

	// Print the data types of the columns
	fmt.Println(df.Types())

	// Print the summary statistics of the dataframe
	fmt.Println(df.Describe())

	// Print first 10 rows of the dataframe
	fmt.Println(df)
}

func (d *DataFrameDataset) ExtractColumns(df dataframe.DataFrame, columnNames []string) dataframe.DataFrame {
	return df.Select(columnNames)
}

func (d *DataFrameDataset) GetCorrelationMatrix(df dataframe.DataFrame) CorrelationMatrix {
	columns := df.Names()
	corrMatrix := make([][]float64, len(columns))
	for i := 0; i < len(columns); i++ {
		corrMatrix[i] = make([]float64, len(columns))
		for j := i; j < len(columns); j++ {
			if i == j {
				corrMatrix[i][j] = 1.0
				continue
			}
			col1 := df.Col(columns[i]).Float()
			col2 := df.Col(columns[j]).Float()
			corr := stat.Correlation(col1, col2, nil)
			corrMatrix[i][j] = corr
		}
	}
	return corrMatrix
}

func (d *DataFrameDataset) ToFloatMatrix(df dataframe.DataFrame) [][]float64 {
	rows := df.Nrow()
	cols := df.Ncol()

	matrix := make([][]float64, cols)
	for i := 0; i < cols; i++ {
		matrix[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			val := df.Elem(j, i).Float()
			matrix[i][j] = val
		}
	}
	return matrix
}

func (corrMat *CorrelationMatrix) Print(headers []string) {
	table := tablewriter.NewWriter(os.Stdout)
	tableHeader := append([]string{""}, headers...)
	table.SetHeader(tableHeader)
	for index, row := range *corrMat {
		stringRow := make([]string, len(row)+1)
		for i, val := range row {
			if i == 0 {
				stringRow[i] = headers[index]
			}
			stringRow[i+1] = fmt.Sprintf("%0.2f", val)
		}
		table.Append(stringRow)
	}
	table.Render()
}
