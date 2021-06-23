using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using UnityEngine.UI;
using System;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;
using NumSharp;
using Emgu.CV;

public class Detector : MonoBehaviour
{
    public InferenceSession session = new InferenceSession(@"Assets/Model/best.onnx");
    public List<NamedOnnxValue> container = new List<NamedOnnxValue>();
    public UnityEngine.UI.Image Imszz;
    public VideoLoader VideoLoader;

    public static int modelW = 416;
    public static int modelH = 416;

    void Start()
    {
        using SixLabors.ImageSharp.Image imageOrg = SixLabors.ImageSharp.Image.Load("Assets/tes.jpg", out IImageFormat format);
        Tensor<float> input = PreProcessing(imageOrg);
        container.Add(NamedOnnxValue.CreateFromTensor("images", input));

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(container);

        var resultsArray = results.ToArray();
        Tensor<float> pred = resultsArray[0].AsTensor<float>();
        var output = PostProcessing(pred);

        var box = DrawBox(output, input, imageOrg);

        var hasil = Image2Array(imageOrg);

    }

    void Update()
    {
        
    }

    public static NDArray Image2Array(SixLabors.ImageSharp.Image input)
    {
        NDArray arrImg = np.zeros((input.Height, input.Width, 3));
        var clone = new Image<Rgb24>(input.Width, input.Height);
        clone.Mutate(i => i.Fill(SixLabors.ImageSharp.Color.Gray));
        clone.Mutate(o => o.DrawImage(input, new Point(0, 0), 1f));

        for(int y=0; y<input.Height; y++){
            Span<Rgb24> pixelSpan = clone.GetPixelRowSpan(y);
            for(int x=0; x<input.Width; x++){
                arrImg[y, x, 0] = pixelSpan[x].B;
                arrImg[y, x, 1] = pixelSpan[x].G;
                arrImg[y, x, 2] = pixelSpan[x].R;
            }
        }
        return arrImg;
    }

    public static int Array2Image(NDArray input)
    {
        //Image<Rgb24> output = new Image<Rgb24>(input.shape[1], input.shape[0]);
        //SixLabors.ImageSharp.Image output = new SixLabors.ImageSharp.Image.Load(input);
        return 0;
    }

    public static Tensor<float> PreProcessing(SixLabors.ImageSharp.Image imageOrg)
    {
        var iw = imageOrg.Width;
        var ih = imageOrg.Height;
        var w = modelW;
        var h = modelH;

        float width = (float)w / iw;
        float height = (float)h / ih;

        float scale = Math.Min(width, height);

        var nw = (int)(iw * scale);
        var nh = (int)(ih * scale);

        var pad_dims_w = (w - nw) / 2;
        var pad_dims_h = (h - nh) / 2;

        var image = imageOrg.Clone(x => x.Resize((nw), (nh)));

        var clone = new Image<Rgb24>(w, h);
        clone.Mutate(i => i.Fill(SixLabors.ImageSharp.Color.Gray));
        clone.Mutate(o => o.DrawImage(image, new Point(pad_dims_w, pad_dims_h), 1f)); // draw the first one top left

        Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, h, w });
        for (int y = 0; y < clone.Height; y++)
        {
            Span<Rgb24> pixelSpan = clone.GetPixelRowSpan(y);
            for (int x = 0; x < clone.Width; x++)
            {
                input[0, 0, y, x] = pixelSpan[x].B / 255f;
                input[0, 1, y, x] = pixelSpan[x].G / 255f;
                input[0, 2, y, x] = pixelSpan[x].R / 255f;
            }
        }

        return input;
    }

    public static NDArray PostProcessing(Tensor<float> pred)
    {
        var prediction = np.array(pred.ToArray()).reshape(1,2535,7);

        bool[,] xt = new bool[1,prediction.shape[1]];

        var nc = prediction.shape[2] - 5;

        int count1 = 0;
        for(int a=0; a < prediction.shape[0]; a++){
            for(int b=0; b < prediction.shape[1]; b++){
                for(int c=0; c < prediction.shape[2]; c++){
                    if(c == 4){
                        if(prediction[a][b][c] > 0.25f){
                            xt[a, b] = true;
                            count1++;
                        }
                    }
                }
            }
        }
        var xc = np.array(xt);

        int max_nms = 30000;
        var output = np.zeros((6));

        for(int i=0; i<prediction.shape[0]; i++){
            var x_ = np.array(prediction[i]);

            List<int> index = new List<int>();

            for(int a=0; a<xc.shape[0]; a++){
                for(int b=0; b<xc.shape[1]; b++){
                    if(xc[a, b] == true){
                        index.Add(b);
                    }
                }
            }

            int[] indexArr = index.ToArray();

            var x = np.zeros((x_.shape[1]));
            for(int a=0; a<indexArr.Length; a++){
                x = np.vstack(x, x_[indexArr[a]]);
            }
            x = x[new Slice(1, x.shape[0])];
            
            if(x.shape[0] == 0){
                continue;
            }

            x[":, 5:"] = x[":, 5:"] * x[":, 4:5"];

            var inp = x[":, :4"];
            var box = np.copy(inp);
            box[":, 0"] = inp[":, 0"] - inp[":, 2"] / 2;
            box[":, 1"] = inp[":, 1"] - inp[":, 3"] / 2;
            box[":, 2"] = inp[":, 0"] + inp[":, 2"] / 2;
            box[":, 3"] = inp[":, 1"] + inp[":, 3"] / 2;
            
            var conf = x[":, 5:"].max(1).reshape((1,1));
            var j = np.argmax(x[":, 5:"], 1).reshape((1,1));
            var xf = np.concatenate((box, conf, j), 1);
            output = np.vstack(output, xf);
        }
        output = output[new Slice(1, output.shape[0])];
        return output;
    }

    public static NDArray DrawBox(NDArray pred, Tensor<float> inp, SixLabors.ImageSharp.Image img)
    {
        var input = np.array(inp.ToArray()).reshape(1,3,modelH,modelW);
        
        var img1_w = input.shape[3];
        var img1_h = input.shape[2];
        var img0_shape = np.array(new int[] {img.Height, img.Width});

        for(int i=0; i<pred.shape[0]; i++){
            var det = pred[i];
            var coords = det[":, :4"];

            var gain = Math.Min(img1_h / img0_shape[0], img1_w / img0_shape[1]);
            var pad = np.array(new double[] {(img1_w - img0_shape[1] * gain) / 2, (img1_h - img0_shape[0] * gain) / 2});

            
            coords[":", new Slice(0, 2)] -= pad[0];
            coords[":", new Slice(1, 3)] -= pad[1];
            coords[":, :4"] /= gain;

            coords[":, 0"] = np.clip(coords[":, 0"], 0, img0_shape[1]);
            coords[":, 1"] = np.clip(coords[":, 1"], 0, img0_shape[0]);
            coords[":, 2"] = np.clip(coords[":, 2"], 0, img0_shape[1]);
            coords[":, 3"] = np.clip(coords[":, 3"], 0, img0_shape[0]);

            det[":, :4"] = coords;
            pred[i] = det;
        }
        return pred;
    }
}
