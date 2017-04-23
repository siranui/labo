object cifar10 {
  import breeze.linalg._
  import math._

  val rand = util.Random
  rand.setSeed(0)

  var W_af1 = new DenseMatrix[Double](900,100)
  var b_af1 = new DenseVector[Double](100)
  var W_af2 = new DenseMatrix[Double](100,10)
  var b_af2 = new DenseVector[Double](10)

  var H = DenseVector.zeros[Double](9).map(a => rand.nextGaussian * 2 - 1) // フィルタ3x3
  var H2 = DenseVector.zeros[Double](9).map(a => 1 - rand.nextGaussian * 2) // フィルタ3x3
  var b = 1d // バイアス
  var mu = 0.3 // 学習率

  /////////////////////////////////////

  class Convolution(var H:DenseVector[Double]) {// {{{
    //var H0 = H
    var X0 = new DenseVector[Double](0)
    var W = new DenseMatrix[Double](0,0)

    def forward(X:List[DenseVector[Double]]) = {
      for(i <- 0 until X.size){
        if(i == 0) {
          W = make_W(X(i),H)
          X0 = X(i)
        }
        else {
          W = DenseMatrix.horzcat(W,make_W(X(i),H))
          X0 = DenseVector.vertcat(X0,X(i))
        }
      }
      //W = make_W(X(0),H)
      //val y = W * X(0) :+ b
      val y = W * X0 :+ b
      y
    }

    def backward(d:DenseVector[Double]) = {
      //val W = make_W(X0,H)
      //val dW = d * X0(0).t
      val dW = d * X0.t
      val dH = DenseVector.zeros[Double](H.size)
      for(i <- 0 until dH.size){
        dH(i) = (Tr(i,W,H) :* dW).sum
      }
      b = b - mu * sum(d)/d.size
      H = H + mu :* dH
      W.t * d
    }

    // Wを作る。 X:流れてきた画像データ H:フィルタ
    def make_W(X:DenseVector[Double], H:DenseVector[Double]): DenseMatrix[Double] = {

      val wx = sqrt(X.size).toInt
      val wh = sqrt(H.size).toInt
      var W = DenseMatrix.zeros[Double](pow(wx-wh+1,2).toInt, X.size)

      for(i <- 0 until pow(wx-wh+1,2).toInt){
        var flag = H.size
        for(j <- 0 until X.size){
          if(i < wx-wh+1 && j%wx+(wx-wh) < wx && flag > 0){
            W(i,j+i%(wx-wh+1)+wx*(i/wh)) = H(H.size - flag)
            flag -= 1
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


  class Affine(var W:DenseMatrix[Double], var b:DenseVector[Double]) {
    var X0: DenseVector[Double] = null
    def forward(X:DenseVector[Double]) = {
      X0 = X.copy
      (X.t * W).t + b
    }

    def backward(d:DenseVector[Double]) = {
      val dW = DenseMatrix(X0).t * DenseMatrix(d)
      val dX = (d.t * W.t).t
      val alpha = 0.05

      b -= d :* alpha
      W += dW :* alpha

      dX
    }
  }

  def read(file:String): List[DenseVector[Double]] = {
    // TODO: take(100)を無くす。(メモリ不足のため途中でtake(100)している)
    // val r = io.Source.fromFile(file).getLines.toList.map(_.split(',').toList.map(_.toDouble))
    val r = io.Source.fromFile(file).getLines.toList.take(100).map(_.split(',').toList.map(_.toDouble))

    var list = List[DenseVector[Double]]()
    for(i <- 0 until r.size){
      var li = DenseVector.zeros[Double](r(0).size)
      for(j <- 0 until r(0).size){
        li(j) = r(i)(j)
      }
      list = list ++ List(li)
    }
    list
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
    W_af1 = W_af1.map(a => rand.nextGaussian - 0.5)
    b_af1 = b_af1.map(a => rand.nextGaussian - 0.5)
    W_af2 = W_af2.map(a => rand.nextGaussian - 0.5)
    b_af2 = b_af2.map(a => rand.nextGaussian - 0.5)

    val conv1 = new Convolution(H)
    val af1 = new Affine(W_af1,b_af1)
    val af2 = new Affine(W_af2,b_af2)
    val r1 = new ReLU()
    val r2 = new ReLU()

    var E = 0d //-1 * sum(correct :* y.map(log))
    var cnt = 0

    // メモリ不足のため100個分のデータを読み込んでいる
    val data = read("/home/yuya/labo/cifar10/data/train-d1.txt")
    val List(labels) = read("/home/yuya/labo/cifar10/data/train-t1.txt")
    // r,g,b に分ける。(家)
    val (r,g,b) = (data.map(_(0 until 1024)),data.map(_(1024 until 2048)),data.map(_(2048 until 3072)))

    do{
      E = 0d
      println(s"--- ${cnt} ---")

      for(i <- 0 until 30){ //TODO: 30 -> data.size
        // forward
        var y = af2.forward(r2.forward(af1.forward(r1.forward(conv1.forward(List(r(i),g(i),b(i)))))))

        // softmax
        val under = sum(y.map(exp))
        y = y.map( a => exp(a) / under )

        val correct = one_of_k(labels(i).toInt)

        // E = -\sum_ (t_k \log y_k: cross entropy(softmax)
        E += -1 * sum(correct :* y.map(log))
        // dE: Eの偏微分
        val dE = y - correct

        // backward
        conv1.backward(r1.backward(af1.backward(r2.backward(af2.backward(dE)))))
      }
      cnt += 1
      println(E)
    }while(E > 50 && cnt < 5)
  }

  ////////////////////////////////////////////////////////

  def main(args: Array[String]) {

    // 入力画像の代わり
    // val fake = DenseVector.zeros[Double](32*32).map(a => rand.nextGaussian)
    // println(conv1.forward(fake))

    // メモリ不足のため100個分のデータを読み込んでいる
    // val data = read("/home/yuya/labo/cifar10/data/train-d1.txt")
    // println(conv1.forward(data(20)(0 until 1024)))

    train()

    println("\nOk.\n")
  }
}
