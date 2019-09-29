using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace powermeterml
{
    public class PowerMeterSpikePrediction
    {
        [VectorType(3)]
        public double[] MeterPrediction { get; set; }
    }
}
