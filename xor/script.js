// ---- Settings ---- //

// model

var inputLayerSize  = 3;
var outputLayerSize = 1;
var hiddenLayerSize = [ 4 ];

var mutationRate = 100;
var batchSize = 10;

// the test cases to run the AI on
var tests = [
    [ 0,0,0,  0 ],
    [ 1,0,0,  1 ],
    [ 0,1,0,  1 ],
    [ 1,1,0,  0 ],
    [ 0,0,1,  1 ],
    [ 1,0,1,  0 ],
    [ 0,1,1,  0 ],
    [ 1,1,1,  1 ],
];

var neural_fn = q => 1/(1+Math.E**-q);

// visualzation

var scale = 10;

var xStretch   =scale* 150,
    yStretch   =scale* 18,
    lineWidth  =scale* 1,
    nodeRadius =scale* 3;

// ---- Matrix ---- //

/**
 * @class
 * @param {number} width the width of the matrix
 * @param {number} height the height of the matrix
 * @param {number|(number,number)=>number|number[][]} fillValue the default value to fill the matrix with or the default matrix content
 * @returns {Matrix}
 */
function Matrix(width,height,fillValue=0) {
    let arr = 
        Array.isArray(fillValue) 
            ? fillValue
            : Array(width).fill().map((_,y)=>Array(height).fill().map((_,x)=>typeof fillValue == 'function' ? fillValue(x,y) : fillValue));
    
    this.width = width;
    this.height = height;

    /**
     * 
     * @returns {number[][]}
     */
    this.toJSON = function() { return arr; }

    return this;
}
Matrix.from = function(arr) { return new Matrix(arr.length,Math.max(...arr.map(r=>r.length)),arr) }

/**
 * Maps a function to each row of the matrix and return a new Matrix without modifying the original one
 * @param {(number[],number,number[][])=>number[]} fn 
 * @returns {Matrix}
 */
Matrix.prototype.map = function(fn) {
    let arr = this.toJSON();
    return new Matrix(this.width,this.height,(y,x)=>fn(arr[y],y,arr))
}

/**
 * @param {Matrix} other 
 */
Matrix.prototype.add = function(other) {
    let nw = Math.min(width,other.width);
    let nh = Math.min(height,other.height);
    let arr = this.toJSON();
    return new Matrix(nw,nh,(c,r)=>arr[r][c]+other.row(r)[c]);
}

Matrix.prototype.column = function(col) { return this.toJSON().map(l=>l[col]); }
Matrix.prototype.row = function(row) { return this.toJSON()[row]; }

// ---- AI Setup ---- //

let randomValue = () => (Math.random()*2-1) * mutationRate;

/**
 * @param {Matrix} inp The input layer
 * @param {Matrix[]} W The weights
 * @param {number[][]} B The biases
 * @returns {number[]} The output layer
 */
function getOutput(inp,W,B) {
    let l = inp.row(0);
    for (let li = 0; li < W.length; li++) {
        l = Array(hiddenLayerSize.concat(outputLayerSize)[li]).fill().map(
            (_,i) => neural_fn(
                W[li].column(i)
                .map((w,j)=>l[j]*w)
                .reduce(
                    (a,b) => a+b,
                    0
                ) + B[li][i]
            )
        );
    }
    return l;
}

/**
 * @param {number[]} tests 
 * @param {Matrix[]} W 
 * @param {number[][]} B 
 * @returns {number} The cummulated error
 */
function getError(tests,W,B) {
    let e = 0;
    for (let test of tests) {
        let inp = Matrix.from([test.slice(0,inputLayerSize)]),
            exp = test.slice(-outputLayerSize);
        let val = getOutput(inp,W,B);
        e += exp.map((e,i)=>Math.abs(e-val[i])).reduce((a,b)=>a+b,0);
    }
    return e;
}

let models = Array(batchSize).fill().map(
    () => ({
        // weights
        W : [
            // input weights
            new Matrix(inputLayerSize,hiddenLayerSize[0],randomValue),
            // neuron weights
            ...hiddenLayerSize.slice(1).map(
                (s,i) => new Matrix(hiddenLayerSize[i],s,randomValue)
            ),
            // output matrix
            new Matrix(hiddenLayerSize.at(-1),outputLayerSize,randomValue)
        ],

        // biases
        B : hiddenLayerSize.map(s=>Array(s).fill().map(randomValue)).concat([Array(outputLayerSize).fill(0)])
    })
);

let model = models[0]; // just pick any model from the list

/**
 * Trains the model once, finds the best model in the batch and creates slightly modified versions of it for the next batch
 */
const train = () => {
    let errors = models.map(m=>getError(tests,m.W,m.B)); // gets the error for all the models in the batch
    let bestValue = Math.min(...errors);                 // finds the best error
    let bestValueIndex = errors.indexOf(bestValue);      // finds the best error index
    let bestModel = models[bestValueIndex];              // gets the best model
        model = bestModel;
    models = Array(batchSize-1).fill().map( // creates variations of the best model for the next batch
        () => ({
            // weights
            W : bestModel.W.map(mat=>Matrix.from(mat.toJSON().map(row=>row.map(e=>e+randomValue())))),
    
            // biases
            B : bestModel.B.map(row=>row.map(e=>e+randomValue())) // bestModel.B.slice(0,-1).map(row=>row.map(e=>e+randomValue())).concat([Array(outputLayerSize).fill(0)])
        })
    ).concat([model]);
}

function runTests(tests) {
    /** @type {{inp:number[][],exp:number[][],val:number[][]}} */
    let dat = {
        inp: [],
        exp: [],
        val: []
    }
    if (model) for (let test of tests) {
        let inp = Matrix.from([test.slice(0,inputLayerSize)]), // input layer
            exp = test.slice(-outputLayerSize),                // expected output
            val = getOutput(inp,model.W,model.B);              // actual output
        dat.inp.push(inp.row(0));
        dat.exp.push(exp);
        dat.val.push(val);
    }
    return dat;
}

// Starts training

setInterval(
    train,
    0
)

// ---- Render the network ---- //

const canvas = document.getElementById('canvas');
/**@type {CanvasRenderingContext2D}*/
const ctx = canvas.getContext('2d');

const render = () => {
    let networkHeight = Math.max(...hiddenLayerSize,inputLayerSize);
    let networkWidth  = hiddenLayerSize.length +2;

    canvas.width  = networkWidth * nodeRadius * 2 + (networkWidth-1) * xStretch;
    canvas.height = (networkHeight-2) * nodeRadius * 2 + (networkHeight-1) * yStretch;

    // HTML stuff

    document.getElementById('error-value').innerText = getError(tests,model.W,model.B).toFixed(19);

    let td = `<tr><td>Input</td><td>Expected</td><td>Network</td></tr>`;

    let {inp,exp,val} = runTests(tests);

    for (let i = 0; i < tests.length; i++) {
        td += `<tr><td>${inp[i]}</td><td>${exp[i]}</td><td style="color:rgb(${255/Math.abs(exp[i]-val[i])},0,0)">${val[i].map(v=>v.toFixed(3))}</td></tr>`;
    }

    document.getElementById('data-table').innerHTML = td;

    // the synapses (weights)

    let mw = Math.max(...model.W.map(ww=>ww.toJSON()).flat(2).map(ww=>Math.abs(ww)));

    for (let ci in model.W) {
        ci=+ci;
        let weights = model.W[ci];
        // each row corresponds to a node on the left and all the elements are all corresponding to a node on the right
        for (let ni in weights.toJSON()) {
            ni=+ni;
            let node = weights.row(ni);
            for (let wi in node) {
                wi=+wi;
                let w = node[wi];
                ctx.lineWidth = lineWidth;
                let dw = w/mw*255;
                ctx.strokeStyle = dw < 0 ? `rgb(0,0,${-dw})` : `rgb(${dw},0,0)`;
                ctx.beginPath();
                ctx.moveTo( 4*nodeRadius+ci*xStretch, 2*nodeRadius+ni*yStretch );
                ctx.lineTo( (ci+1)*xStretch, 2*nodeRadius+wi*yStretch );
                ctx.stroke();
            }
        }
    }

    // all the nodes (biases)

    let mb = Math.max(...model.B.flat());

    // input layer
    for (let i = 0; i < inputLayerSize; i++) {
        ctx.fillStyle = `rgb(0,200,100)`;
        ctx.beginPath();
        ctx.arc(
            2*nodeRadius, 2*nodeRadius+i*yStretch,
            2*nodeRadius,
            0, Math.PI*2
        );
        ctx.fill();
    }

    for (let ci in model.B) {
        ci=+ci;
        let biases = model.B[ci];
        for (let bi in biases) {
            bi=+bi;
            let bias = biases[bi];
            let db = bias/mb*255;
            ctx.fillStyle = db < 0 ? `rgb(0,0,${-db})` : `rgb(${db},0,0)`;
            ctx.beginPath();
            ctx.arc(
                2*nodeRadius+(ci+1)*xStretch,2*nodeRadius+bi*yStretch,
                2*nodeRadius,
                0, Math.PI*2
            );
            ctx.fill();
        }
    }
}

setInterval(
    render,
    100
)
render();