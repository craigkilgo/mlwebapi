using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Text;

namespace webapi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PhishController : ControllerBase
    {
        // GET api/values
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            return new string[] { "not a phish" };
        }

        // GET api/values/5
        [HttpGet("{id}")]
        public ActionResult<string> Get(int id)
        {
            return "value";
        }

        // POST api/values
        [HttpPost]
        public IActionResult PostJsonString([FromBody] string text)
        {

            string modelPath = Path.Combine(Environment.CurrentDirectory, ".", "Model.zip");
            MLContext mlContext = new MLContext(seed: 0);
            ITransformer loadedModel;

            using (var stream = System.IO.File.OpenRead(modelPath))
                loadedModel = mlContext.Model.Load(stream);

            //Predict(mlContext, loadedModel,text);

            Response res = new Response(text);
            res.Predict(mlContext,loadedModel);
            
            //res.Prediction = "Not a phish";
            //res.Probability = 0.030;

            return Ok(res);
            //return text;
        }








        // PUT api/values/5
        [HttpPut("{id}")]
        public void Put(int id, [FromBody] string value)
        {
        }

        // DELETE api/values/5
        [HttpDelete("{id}")]
        public void Delete(int id)
        {
        }


    }

        public class Response
        {
            public Response(string u){
                this.Url=u;
            }
            public string Url;
            public string Prediction;
            public double Probability;
            public void Predict(MLContext mlContext, ITransformer model)
            {

                var predictionFunction = model.MakePredictionFunction<PhishData, PhishPrediction>(mlContext);

                PhishData sample = new PhishData
                {
                    UrlText = Url
                };

                var resultprediction = predictionFunction.Predict(sample);
                this.Prediction = resultprediction.Prediction ? "Phish" : "Not Phish";
                this.Probability = resultprediction.Probability;

                /* Console.WriteLine();
                Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

                Console.WriteLine();
                Console.WriteLine($"Url: {sample.UrlText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Phish" : "Not Phish")} | Probability: {resultprediction.Probability} ");

                Console.WriteLine("=============== End of Predictions ===============");
                Console.WriteLine();*/
                

            }
        }
}
