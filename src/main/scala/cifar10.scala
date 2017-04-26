object cifar10 {
  import breeze.linalg._
  import math._

  val rand = new util.Random(0)

  val xw = 32
  val hw = 3

  val DATA_SIZE = xw*xw
  val FILTER_SIZE = hw*hw
  val AFTER_CONV_SIZE = pow( xw - hw + 1 , 2 ).toInt //ex: (32-3+1)^2=30*30=900

  var READ_SIZE = 200
  var TEST_SIZE = 1000
  var TRAINING_EPOCH = 100

  val HIDDEN = 100
  val OUT = 10

  var W_af1 = new DenseMatrix[Double](AFTER_CONV_SIZE,HIDDEN)
  var b_af1 = new DenseVector[Double](HIDDEN)
  var W_af2 = new DenseMatrix[Double](HIDDEN,OUT)
  var b_af2 = new DenseVector[Double](OUT)

  var H1 = DenseVector.fill(FILTER_SIZE){rand.nextGaussian * 0.01}
  var H2 = DenseVector.fill(FILTER_SIZE){rand.nextGaussian * 0.01}
  var H3 = DenseVector.fill(FILTER_SIZE){rand.nextGaussian * 0.01}
  var H = Array(H1,H2,H3)

  var b = 0d // バイアス
  var mu = 0.03 // 学習率

  var flag_f = List(false,false,false,false) //(conv, affine, sig, relu)
  var flag_b = List(false,false,false,false) //(conv, affine, sig, relu)


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
        X0(i) = X(i)
        W(i) = make_W(H(i))
        u += W(i) * X(i)
      }
      val y = u :+ b

      //debug
      if(flag_f(0)){
        println(s"conv.forward.W(0).max = ${W(0).max}")
        println(s"conv.forward.W(0).min = ${W(0).min}")
        println(s"conv.forward.W(0).sum = ${W(0).sum}")
        println(s"conv.forward.W(0)(0 until 2, 0 until 3) =\n${W(0)(0 until 2, 0 until 3)}\n")
        println(s"conv.forward.b = ${b}\n")
        println(s"conv.forward.y.max = ${y.max}")
        println(s"conv.forward.y.min = ${y.min}")
        println(s"conv.forward.y.sum = ${y.sum}")
        println(s"conv.forward.y(0 until 10) = ${y(0 until 10)}\n")
      }

      y
    }

    def backward(d:DenseVector[Double]) = {
      // debug
      if(flag_b(0)){
        println(s"conv.backward.H(0) = ${H(0)}")
        println(s"conv.backward.H(1) = ${H(1)}")
        println(s"conv.backward.H(2) = ${H(2)}\n")
      }

      var dW = new Array[DenseMatrix[Double]](H.size)
      for(hi <- 0 until H.size){
        //var dW = d * X0(hi).t
        dW(hi) = d * X0(hi).t
        for(i <- 0 until xw-hw+1){
          for(j <- 0 until xw-hw+1){
            for(p <- 0 until hw){
              for(q <- 0 until hw){
                H(hi)(p*hw+q) -= mu * dW(hi)(i*(xw-hw+1)+j,(i+p)*xw+j+q)
              }
            }
          }
        }
      }

      // debug
      if(flag_b(0)){
        println(s"conv.backward.dW(0).max = ${dW(0).max}")
        // println(s"conv.backward.dW(0).min = ${dW(0).min}")
        // println(s"conv.backward.dW(0).sum = ${dW(0).sum}")
        println(s"conv.backward.dW(1).max = ${dW(1).max}")
        // println(s"conv.backward.dW(1).min = ${dW(1).min}")
        // println(s"conv.backward.dW(1).sum = ${dW(1).sum}")
        println(s"conv.backward.dW(2).max = ${dW(2).max}")
        // println(s"conv.backward.dW(2).min = ${dW(2).min}")
        // println(s"conv.backward.dW(2).sum = ${dW(2).sum}\n")
        // println(s"conv.backward.H(0) = ${H(0)}")
        // println(s"conv.backward.H(1) = ${H(1)}")
        // println(s"conv.backward.H(2) = ${H(2)}\n")
      }

      b = b - mu * sum(d)
      // W(0).t * d
    }

    // Wを作る  H:フィルタ
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
      //debug
      if(flag_b(3)){
        println(s"relu.backward.x1.map(relu_deriv) = ${x1.map(relu_deriv)}\n")
        println(s"relu.backward.d = ${d}\n")
      }

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
      //debug
      if(flag_b(2)){
        println(s"sigmoid.backward.d = ${d}\n")
      }

      d :* y0 :* (1d:-y0)
    }
  }

  class Affine(var W_af:DenseMatrix[Double], var b_af:DenseVector[Double]) {
    var X0: DenseVector[Double] = null
    def forward(X:DenseVector[Double]) = {
      X0 = X.copy
      (X.t * W_af).t + b_af
    }

    def backward(d:DenseVector[Double]) = {
      //debug
      if(flag_f(1)){
        println(s"affine.backward.d = ${d}\n")
      }

      val dW = DenseMatrix(X0).t * DenseMatrix(d)
      val dX = (d.t * W_af.t).t

      b_af -= d :* mu
      W_af -= dW :* mu

      dX
    }
  }

  def softmax(y:DenseVector[Double]) =  {
    val m = y.max
    val under = sum(y.map(a=>exp(a-m)))
    y.map( a => exp(a-m) / under )
  }

  def read(file:String) = {
    val r = io.Source.fromFile(file).getLines.toArray.take(READ_SIZE).map(_.split(',').toList.map(_.toDouble))

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

  def read_data__home(file:String) = {
    val r = io.Source.fromFile(file).getLines.toArray.take(READ_SIZE).map(_.split(',').toList.map(_.toDouble/256d))

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

  def read_data__labo(file:String) = {
    val f = io.Source.fromFile(file).getLines.toArray.take(READ_SIZE)

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
    W_af2 = W_af2.map(a => rand.nextGaussian)
    b_af2 = b_af2.map(a => rand.nextGaussian)

    val conv1 = new Convolution()
    val af1 = new Affine(W_af1,b_af1)
    val af2 = new Affine(W_af2,b_af2)
    val f1 = new Sigmoid()
    val f2 = new ReLU()

    var E = 0d
    val eps = 0.00001 //log(0) = -Infinity, log(-i) = NaN (i>0) を避けるため
    var cnt = 0

    val data = read_data__home("/home/yuya/labo/cifar10/data/train-d1.txt")
    val Array(labels) = read("/home/yuya/labo/cifar10/data/train-t1.txt")

    // r,g,b に分ける
    val (r,g,b) = (data.map(_(0 until 1024)),data.map(_(1024 until 2048)),data.map(_(2048 until 3072)))

    //debug
    /* println(s"\nH1 = ${H1}\nH2 = ${H2}\nH3 = ${H3}") */

    do{
      E = 0d
      println(s"--- ${cnt} ---")

      for(i <- 0 until data.size){
        //debug
        if(i%(READ_SIZE/5)==0) println(s"now training... (${i}/${data.size})")

        // forward net
        var y = af2.forward(f2.forward(af1.forward(f1.forward(conv1.forward(Array(r(i),g(i),b(i)))))))
        //var y = af2.forward(af1.forward(f1.forward(conv1.forward(Array(r(i),g(i),b(i))))))
        //var y = f1.forward(af1.forward(conv1.forward(Array(r(i),g(i),b(i)))))


        // softmax
        y = softmax(y)

        val correct = one_of_k(labels(i).toInt)

        // E = -\sum_ (t_k \log y_k: cross entropy(softmax)
        E += -1 * sum(correct :* y.map(a=>log(a+eps)))
        // dE: Eの偏微分
        val dE = y - correct

        // backward net
        conv1.backward(f1.backward(af1.backward(f2.backward(af2.backward(dE)))))
        //conv1.backward(f1.backward(af1.backward(af2.backward(dE))))
        //conv1.backward(af1.backward(f1.backward(dE)))

        //debug
        /* if(i%(READ_SIZE/5)==0) println(s"E${i}: ${E}") */
      }
      cnt += 1
      println(s"E = ${E}")

      //debug
      println(s"\nH1 = ${H1}\nH2 = ${H2}\nH3 = ${H3}\nb = ${cifar10.b}")
      /* println(s"\nW1 = ${W_af1}\n b1=${b_af1}") */
    }while(E > READ_SIZE * 2 && cnt < TRAINING_EPOCH)

    println("finish training\n")
    E
  }

  ////////////////////////////////////////////////////////

  def ask(target:DenseVector[Double], correct:Int) = {
    val conv1 = new Convolution()
    val af1 = new Affine(W_af1,b_af1)
    val af2 = new Affine(W_af2,b_af2)
    val f1 = new Sigmoid()
    val f2 = new ReLU()

    val r = target(0 until 1024)
    val g = target(1024 until 2048)
    val b = target(2048 until 3072)

    // forward net
    var y = af2.forward(f2.forward(af1.forward(f1.forward(conv1.forward(Array(r,g,b))))))
    //var y = af2.forward(af1.forward(f1.forward(conv1.forward(Array(r,g,b)))))
    //var y = f1.forward(af1.forward(conv1.forward(Array(r,g,b))))

    y = softmax(y)

    println(s"y: $y")

    val predict_number = argmax(y) //yの最大値のある要素番号を返す

    println(s"predict: ${predict_number}, correct: ${correct}")

    if(predict_number == correct) 1d
    else 0d
  }

  def test() = {
    val tmp = READ_SIZE
    READ_SIZE = TEST_SIZE

    val data = read_data__home("/home/yuya/labo/cifar10/data/test-d.txt")
    val Array(labels) = read("/home/yuya/labo/cifar10/data/test-t.txt")

    READ_SIZE = tmp

    flag_f.map(a=>false)

    var correct = 0d
    for(i <- 0 until data.size){
      correct += ask(data(i),labels(i).toInt)
    }

    val correct_rate = 100d * correct / data.size
    println(s"correct rate: ${correct_rate}%")

    correct_rate
  }


  ////////////////////////////////////////////////////////

  def main(args: Array[String]) {

    val E = train()
    val correct_rate = test()

    println(s"Filter: ${hw}*${hw}; Affine: A1(${AFTER_CONV_SIZE}->${HIDDEN}),A2(${HIDDEN}->${OUT}); Train Rate: ${mu}; Train Loop: ${TRAINING_EPOCH}; Read Size: ${READ_SIZE}; Correct Rate: ${correct_rate}; E: ${E}")
    println(s"\nH1 = ${H1}\nH2 = ${H2}\nH3 = ${H3}\nb = ${cifar10.b}")

    println("\nOk.\n")
  }
}
