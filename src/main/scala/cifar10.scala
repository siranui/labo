object cifar10 {
  import breeze.linalg._
  import math._

  val rand = new util.Random(0)

  val xw = 32
  val hw = 3

  val DATA_SIZE = xw*xw
  val FILTER_SIZE = hw*hw
  val AFTER_CONV_SIZE = pow( xw - hw + 1 , 2 ).toInt //ex: (32-3+1)^2=30*30=900

  var READ_SIZE = 500

  var TRAINING_EPOCH = 10

  var W_af1 = new DenseMatrix[Double](AFTER_CONV_SIZE,10)
  var b_af1 = new DenseVector[Double](10)
  var W_af2 = new DenseMatrix[Double](100,10)
  var b_af2 = new DenseVector[Double](10)

  var H1 = DenseVector.zeros[Double](FILTER_SIZE).map(a => rand.nextGaussian * 0.01)
  var H2 = DenseVector.zeros[Double](FILTER_SIZE).map(a => rand.nextGaussian * 0.01)
  var H3 = DenseVector.zeros[Double](FILTER_SIZE).map(a => rand.nextGaussian * 0.01)
  var H = Array(H1,H2,H3)

  var b = 0d // バイアス
  var mu = 0.01 // 学習率

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  class Convolution() {// {{{
    var X0: Array[DenseVector[Double]] = null
    var W:  Array[DenseMatrix[Double]] = null

    def forward(X:Array[DenseVector[Double]]) = {
      // X0 = new Array[DenseVector[Double]](0)
      // W = new Array[DenseMatrix[Double]](0)
      X0 = new Array[DenseVector[Double]](X.size)
      W = new Array[DenseMatrix[Double]](X.size)

      var u = DenseVector.zeros[Double](AFTER_CONV_SIZE)
      for(i <- 0 until X.size){
        //X0 = X0 ++ Array(X(i))
        //W = W ++ Array(make_W(H(i)))
        X0(i) = X(i)
        W(i) = make_W(H(i))
        u += W(i) * X(i)
      }
      val y = u :+ b
      y
    }

    def backward(d:DenseVector[Double]) = {
      for(hi <- 0 until H.size){
        var dW = d * X0(hi).t
        for(i <- 0 until xw-hw+1){
          for(j <- 0 until xw-hw+1){
            for(p <- 0 until hw){
              for(q <- 0 until hw){
                H(hi)(p*hw+q) -= mu * dW(i*(xw-hw+1)+j,(i+p)*xw+j+q)
              }
            }
          }
        }
      }

      b = b - mu * sum(d)
      W(0).t * d
    }

    // Wを作る。 X:流れてきた画像データ H:フィルタ
    def make_W(H:DenseVector[Double]): DenseMatrix[Double] = {
      var W = DenseMatrix.zeros[Double](pow(xw-hw+1,2).toInt, DATA_SIZE)
      for(i <- 0 until xw-hw+1){
        for(j <- 0 until xw-hw+1){
          for(p <- 0 until hw){
            for(q <- 0 until hw){
              W(i*(xw-hw+1)+j, (i+p)*xw+j+q) = H(p*hw+q)
            }
          }
        }
      }
      W
    }

    def Tr(r:Int, W:DenseMatrix[Double], H:DenseVector[Double]) = {
      W.map(a => if(a == H(r)) 1d else 0d)
    }

  }// }}}

  class ReLU(){
    var x1:DenseVector[Double] = null
    def forward(x:DenseVector[Double])={
      x1 = x.copy
      val y = x.map(relu)
      y
    }

    def relu(x:Double)={
      if(x>0) x else 0d
    }

    def relu_deriv(x:Double)={
      if(x>0) 1d else 0d
    }

    def backward(d:DenseVector[Double])={
      d :* x1.map(relu_deriv)
    }
  }

  class Sigmoid(){
    var y0 :DenseVector[Double] = null
    def forward(x:DenseVector[Double]) = {
      y0 = x.map(a => 1/(1+exp(-a)))
      y0
    }

    def backward(d:DenseVector[Double]) = {
      d :* y0 :* (1d:-y0)
    }
  }

  class Affine(var W:DenseMatrix[Double], var b:DenseVector[Double]) {
    var X0: DenseVector[Double] = null
    def forward(X:DenseVector[Double]) = {
      X0 = X.copy
      (X.t * W).t + b
    }

    def backward(d:DenseVector[Double]) = {
      val dW = DenseMatrix(X0).t * DenseMatrix(d)
      val dX = (d.t * W.t).t
      //val alpha = 0.3

      b -= d :* mu
      W += dW :* mu

      dX
    }
  }

  def softmax(y:DenseVector[Double]) =  {
    val m = y.max
    val under = sum(y.map(a=>exp(a-m)))

    //debug
    //println(s"softmax: y:$y\nm=$m, under:$under")

    y.map( a => exp(a-m) / under )
  }

  def read(file:String) = {
    val r = io.Source.fromFile(file).getLines.toList.take(READ_SIZE).map(_.split(',').toList.map(_.toDouble))

    var list = List[DenseVector[Double]]()
    for(i <- 0 until r.size){
      var li = DenseVector.zeros[Double](r(0).size)
      for(j <- 0 until r(0).size){
        li(j) = r(i)(j)
      }
      list = li :: list
    }
    list.reverse.toArray
  }


  def read_l(file:String) = {
    val f = io.Source.fromFile(file).getLines.toArray.take(READ_SIZE) /* TODO: remove take_size */

    /* labo */
    val ff = f.map(_.split(",").map(_.toDouble/256d).toArray).toArray
    val r = ff.map(a => (0 until a.size by 3).map(k => a(k)).toArray)
    val g = ff.map(a => (1 until a.size by 3).map(k => a(k)).toArray)
    val b = ff.map(a => (2 until a.size by 3).map(k => a(k)).toArray)

    var list = List[DenseVector[Double]]()
    for(i <- 0 until ff.size){
      var li = DenseVector.vertcat(DenseVector.vertcat(DenseVector(r(i)),DenseVector(g(i))), DenseVector(b(i)))

      list = li :: list
    }

    list.reverse.toArray
  }


  def one_of_k(x:Int) = {
    var res : DenseVector[Double] = null
    x.toInt match {
      case 0 => res = DenseVector(1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d)
      case 1 => res = DenseVector(0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d)
      case 2 => res = DenseVector(0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d)
      case 3 => res = DenseVector(0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d)
      case 4 => res = DenseVector(0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d)
      case 5 => res = DenseVector(0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d)
      case 6 => res = DenseVector(0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d)
      case 7 => res = DenseVector(0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d)
      case 8 => res = DenseVector(0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d)
      case 9 => res = DenseVector(0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d)
    }
    res
  }


  ////////////////////////////////////////////////////////

  def train() = {
    // Affine層の重みとバイアスの初期化
    W_af1 = W_af1.map(a => rand.nextGaussian)
    b_af1 = b_af1.map(a => rand.nextGaussian)
    W_af2 = W_af2.map(a => rand.nextGaussian * 5)
    b_af2 = b_af2.map(a => rand.nextGaussian * 5)

    val conv1 = new Convolution()
    val af1 = new Affine(W_af1,b_af1)
    val af2 = new Affine(W_af2,b_af2)
    val f1 = new Sigmoid()
    val f2 = new Sigmoid()

    var E = 0d
    val eps = 0.00001 //log(0) = -Infinity, log(-i) = NaN (i>0) を避けるため
    var cnt = 0

    val data = read("/home/yuya/labo/cifar10/data/train-d1.txt")
    val Array(labels) = read("/home/yuya/labo/cifar10/data/train-t1.txt")

    // r,g,b に分ける
    val (r,g,b) = (data.map(_(0 until 1024)),data.map(_(1024 until 2048)),data.map(_(2048 until 3072)))

    //debug
    println(s"\nH1 = ${H1}\nH2 = ${H2}\nH3 = ${H3}")

    do{
      E = 0d
      println(s"--- ${cnt} ---")

      for(i <- 0 until data.size){
        //debug
        if(i%(READ_SIZE/5)==0) println(s"now training... (${i}/${data.size})")

        // forward
        //var y = af2.forward(f2.forward(af1.forward(f1.forward(conv1.forward(Array(r(i),g(i),b(i)))))))
        var y = f1.forward(af1.forward(conv1.forward(Array(r(i),g(i),b(i)))))


        // softmax
        y = softmax(y)

        val correct = one_of_k(labels(i).toInt)

        // E = -\sum_ (t_k \log y_k: cross entropy(softmax)
        E += -1 * sum(correct :* y.map(a=>log(a+eps)))
        // dE: Eの偏微分
        val dE = y - correct

        // backward
        //conv1.backward(f1.backward(af1.backward(f2.backward(af2.backward(dE)))))
        conv1.backward(af1.backward(f1.backward(dE)))

        //debug
        if(i%(READ_SIZE/5)==0) println(s"E${i}: ${E}")
      }
      cnt += 1
      println(s"E = ${E}")

      //debug
      println(s"\nH1 = ${H1}\nH2 = ${H2}\nH3 = ${H3}\nb = $b")
      println(s"\nW1 = ${W_af1}\n b1=${b_af1}")
    }while(E > READ_SIZE * 1.5 && cnt < TRAINING_EPOCH)

    println("finish training\n")
  }

  ////////////////////////////////////////////////////////

  def ask(target:DenseVector[Double], correct:Int) = {
    val conv1 = new Convolution()
    val af1 = new Affine(W_af1,b_af1)
    val af2 = new Affine(W_af2,b_af2)
    val f1 = new Sigmoid()
    val f2 = new Sigmoid()

    val r = target(0 until 1024)
    val g = target(1024 until 2048)
    val b = target(2048 until 3072)

    // forward
    //var y = af2.forward(f2.forward(af1.forward(f1.forward(conv1.forward(Array(r,g,b))))))
    var y = f1.forward(af1.forward(conv1.forward(Array(r,g,b))))

    y = softmax(y)

    println(s"y: $y")

    val predict_number = y.toArray.indexOf(y.max)

    println(s"predict: ${predict_number}, correct: ${correct}")

    if(predict_number == correct) 1d
    else 0d
  }

  def test() = {
    val data = read("/home/yuya/labo/cifar10/data/test-d.txt")
    val Array(labels) = read("/home/yuya/labo/cifar10/data/test-t.txt")

    var correct = 0d
    for(i <- 0 until data.size){
      correct += ask(data(i),labels(i).toInt)
    }

    println(s"correct rate: ${100d * correct / data.size}%")
  }


  ////////////////////////////////////////////////////////

  def main(args: Array[String]) {

    train()
    test()

    println(s"Read Size: ${READ_SIZE}, Filter: ${hw}*${hw}, Train Rate: ${mu}, Train Loop: ${TRAINING_EPOCH}")

    println("\nOk.\n")
  }
}
