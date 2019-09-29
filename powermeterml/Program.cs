using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace powermeterml
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView dataMeterFileRaw = mlContext.Data.LoadFromTextFile<PowerMeterRecord>("power-export_min.csv", separatorChar: ',', hasHeader: true);

            string outputColumn = nameof(PowerMeterSpikePrediction.MeterPrediction);
            string inputColumn = nameof(PowerMeterRecord.ConsumptionDiffNormalized);

            SsaSpikeEstimator ssaSpikeEstimator = mlContext.Transforms.DetectSpikeBySsa(outputColumn, inputColumn, 95, 30, 70, 30);
            SsaSpikeDetector detector = ssaSpikeEstimator.Fit(dataMeterFileRaw);

            SrCnnAnomalyEstimator ssrAnomalyEstimator = mlContext.Transforms.DetectAnomalyBySrCnn(outputColumn, inputColumn, 30);
            SrCnnAnomalyDetector srCnnAnomalyDetector = ssrAnomalyEstimator.Fit(dataMeterFileRaw);
            
            mlContext.Model.Save(detector, dataMeterFileRaw.Schema, "powermeterspikemodel.zip");
            Console.WriteLine("Saved");

            ITransformer trainedModelFromDisk = mlContext.Model.Load("powermeterspikemodel.zip", out var dataLoader);

            IDataView estimateSpikeData = trainedModelFromDisk.Transform(dataMeterFileRaw);
            IEnumerable<PowerMeterSpikePrediction> powerMeterSpikePredictions = mlContext.Data.CreateEnumerable<PowerMeterSpikePrediction>(estimateSpikeData, false);

            IDataView srCnnAnomaly = srCnnAnomalyDetector.Transform(dataMeterFileRaw);
            IEnumerable<PowerMeterSpikePrediction> powerMeterSrCnnAnomalyPredictions = mlContext.Data.CreateEnumerable<PowerMeterSpikePrediction>(srCnnAnomaly, false);


            var normalizedConsumptionDiff = dataMeterFileRaw.GetColumn<float>(nameof(PowerMeterRecord.ConsumptionDiffNormalized)).ToArray();
            var time = dataMeterFileRaw.GetColumn<DateTime>(nameof(PowerMeterRecord.ConsumptionTime)).ToArray();

            Console.WriteLine("======Displaying anomalies in the Power meter data=========");
            Console.WriteLine("Date              \tReadingDiff\tAlert\tScore\tP-Value");

            StreamWriter textWriterSpikePredictions = new StreamWriter("SpikePredictions.csv");
            textWriterSpikePredictions.WriteLine("Date              \tReadingDiff\tAlert\tScore\tP-Value");

            int i = 0;
            foreach (var spike in powerMeterSpikePredictions)
            {
                if (spike.MeterPrediction[0] == 1)
                {
                    Console.BackgroundColor = ConsoleColor.DarkYellow;
                    Console.ForegroundColor = ConsoleColor.Black;
                }

                Console.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}",
                    time[i], normalizedConsumptionDiff[i],
                    spike.MeterPrediction[0], spike.MeterPrediction[1], spike.MeterPrediction[2]);

                textWriterSpikePredictions.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}",
                    time[i], normalizedConsumptionDiff[i],
                    spike.MeterPrediction[0], spike.MeterPrediction[1], spike.MeterPrediction[2]);

                Console.ResetColor();
                i++;
            }

            Console.WriteLine("======Displaying anomalies in the Power meter data=========");
            Console.WriteLine("Date              \tReadingDiff\tSrCNNAlert\tScore\tP-Value");

            StreamWriter textWriterSrCnnAnomalyPredictions = new StreamWriter("SrCnnAnomalyPredictionsPredictions.csv");
            textWriterSrCnnAnomalyPredictions.WriteLine("Date              \tReadingDiff\tSrCNNAlert\tScore\tP-Value");

            i = 0;
            foreach (var spike in powerMeterSrCnnAnomalyPredictions)
            {
                if (spike.MeterPrediction[0] == 1)
                {
                    Console.BackgroundColor = ConsoleColor.DarkYellow;
                    Console.ForegroundColor = ConsoleColor.Black;
                }

                Console.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}",
                    time[i], normalizedConsumptionDiff[i],
                    spike.MeterPrediction[0], spike.MeterPrediction[1], spike.MeterPrediction[2]);

                textWriterSrCnnAnomalyPredictions.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}",
                    time[i], normalizedConsumptionDiff[i],
                    spike.MeterPrediction[0], spike.MeterPrediction[1], spike.MeterPrediction[2]);

                Console.ResetColor();
                i++;
            }

            textWriterSpikePredictions.Flush();
            textWriterSrCnnAnomalyPredictions.Flush();
        }
    }
}
