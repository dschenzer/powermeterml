using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace powermeterml
{
    public class PowerMeterRecord
    {
        [LoadColumn(0)]
        public string Name { get; set; }
        [LoadColumn(1)]
        public DateTime ConsumptionTime { get; set; }
        [LoadColumn(2)]
        public float ConsumptionDiffNormalized { get; set; }
    }
}
