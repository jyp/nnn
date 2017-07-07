{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UnicodeSyntax #-}

import Prelude hiding (tanh,Num(..),Floating(..))
import qualified Prelude

import Data.List (intercalate)
import GHC.TypeLits
-- import GHC.TypeLits.KnownNat
import Data.Proxy
import Data.Monoid hiding (Last)
import Control.Monad (ap)
------------------
-- TYPES 
type family (++) xs ys where
   '[] ++  xs       = xs
   (x ': xs) ++ ys       = x ': (xs ++ ys)

-- type family (\\) xs y where
--    '[x] \\ x = '[]
--    (x ': xs) \\ y = x ': (xs \\ y)

type family Last xs where
  Last '[x] = x
  Last (x ': xs) = Last xs

type family Reverse' xs ys where
  Reverse' '[] ys = ys
  Reverse' (x ': xs) ys = Reverse' xs (x ': ys )

type family Reverse xs where
  Reverse xs = Reverse' xs '[]

data V (n::Nat) a = V [a]
  deriving (Functor, Foldable, Traversable)

data Typ = Float32 | Int32

instance Show Typ where
  show Float32 = "tf.float32"
  show Int32 = "tf.int32"

type Shape = [Nat]

data T (shape :: Shape) (t :: Typ) where
  T :: String -> T shape t

data SNat (n :: Nat) where
  SNat :: KnownNat n => Proxy n -> SNat n

data SShape s where
  Nil :: SShape '[]
  Cons :: SNat x -> SShape xs -> SShape (x ': xs)

class KnownLen s => KnownShape s where
  shapeSing :: SShape s

instance KnownShape '[] where
  shapeSing = Nil

instance (KnownNat x, KnownShape xs) => KnownShape (x ': xs) where
  shapeSing = Cons (SNat Proxy) shapeSing

class KnownTyp t where
  typVal :: Typ
instance KnownTyp 'Float32 where
  typVal = Float32

instance KnownTyp 'Int32 where
  typVal = Int32

showTyp :: ∀ t. KnownTyp t => String
showTyp = show (typVal @t)


class KnownLen s where
  shapeLen :: Integer

instance KnownLen '[] where
  shapeLen = 0

instance KnownLen xs => KnownLen (x ': xs) where
  shapeLen = 1 Prelude.+ shapeLen @ xs


getShape :: ∀s. KnownShape s=> SShape s
getShape = shapeSing

shapeToList' :: SShape s -> [Integer]
shapeToList' Nil = []
shapeToList' (Cons (SNat x) xs) = natVal x : shapeToList' xs

shapeToList :: ∀(s::Shape). KnownShape s => [Integer]
shapeToList = shapeToList' (getShape @ s)

showShape :: ∀ (s :: Shape). KnownShape s => String
showShape = list (map showDim (reverse (shapeToList @ s)))

showShapeLen :: ∀ (s::Shape). KnownLen s => String
showShapeLen = show (shapeLen @ s)

rememberNat :: SNat n -> (KnownNat n => r) -> r
rememberNat (SNat _) k = k

showDim :: Integer -> String
showDim n = if n == -1 then "None" else show n
--------------------------------
-- Generation Effects

type Effect = Integer -> [String]

newtype Gen x = Gen {fromGen ::  (x -> Effect) -> Effect} deriving (Functor)

instance Applicative Gen where
  pure = return
  (<*>) = ap
  
instance Monad Gen where
  return x = Gen $ \k -> k x
  Gen m >>= f = Gen $ \k -> m $ \a -> fromGen (f a) $ \b -> k b

newVar :: Gen String
newVar = Gen $ \ k n -> k ("var" <> show n) (1 Prelude.+ n)

gen :: String -> Gen ()
gen s = Gen $ \ k n -> s : k () n

type Tensor shape = T shape

-----------------------------------------
-- Generation helpers

parens :: String -> String
parens x = "(" <> x <> ")"

brackets :: String -> String
brackets x = "[" <> x <> "]"

commas :: [String] -> String
commas = intercalate ", "

list :: [String] -> String
list = brackets . commas

(<--) :: ∀ (s :: Shape) t. [Char] -> T s t -> Gen ()
x <-- T y = gen (x <> " = " <> y)

funcall :: String -> [String] -> String
funcall f args = f <> (parens (commas args))

binOp :: ∀ s1 s2 s3 t1 t2 t3. String -> Tensor s1 t1 -> Tensor s2 t2 -> Tensor s3 t3
binOp op (T x) (T y) = T (funcall op [ x , y])

unOp :: ∀ s1 s2 t1 t2. String -> Tensor s1 t1 -> Tensor s2 t2
unOp op (T x) = T (funcall op [x])

assign :: ∀s t. (T s t) -> Gen (T s t)
assign x = do
  v <- newVar
  v <-- x
  return (T v)


--------------------------
-- TF primitives

zeros :: ∀ t (shape :: Shape). KnownShape shape => (T shape t)
zeros = T (funcall "tf.zeros" [(showShape @ shape)])


-- | Declare a parameter to optimize.
parameter' :: ∀ (shape :: Shape) t. String -> T shape t -> Gen (T shape t)
parameter' name (T initial) = do -- FIMXE: initialization function
  v <- newVar
  v <-- T (funcall "tf.Variable" [initial, "name=" <> show name])
  return (T v)

placeholder :: ∀t s. (KnownShape s, KnownTyp t) => String -> Gen (T s t)
placeholder name = do
  gen (name <> " = " <> funcall "tf.placeholder" [showTyp @t,"shape=" <> showShape @ s])
  return (T name)

reduceAll :: String -> Tensor s t -> Tensor '[] t
reduceAll op = unOp ("tf.reduce_" ++ op)

reduceMeanAll :: ∀ (s :: Shape) t. Tensor s t -> Tensor '[] t
reduceMeanAll = reduceAll "mean"

reduce :: ∀ s s' n t. KnownLen s' => String -> Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduce op (T x) = T (funcall ("tf.reduce_" ++ op) [x, "axis=" <> show (shapeLen @ s')])

reduceSum, reduceMean :: ∀ s s' n t. KnownLen s' => Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduceSum = reduce @s @s' @n "sum"
reduceMean = reduce @s @s' @n "mean"

reduceSum0 :: ∀ s' n. KnownLen s' => Tensor (n ': s') 'Float32 -> Tensor s' 'Float32
reduceSum0 = reduceSum @'[]

add :: ∀ d s t. Tensor (d++s) t -> Tensor d t -> Tensor (d++s) t -- note ++s for for 'broadcasting'
add = binOp "tf.add_n"

add_n :: ∀ s t. [Tensor s t] -> Tensor s t
add_n = error "add_n not implemented"

(⊕) :: ∀ (d :: Shape) (s :: Shape) t. Tensor (d ++ s) t -> Tensor d t -> Tensor (d ++ s) t
(⊕) = add @d @s

multiply :: Tensor d t -> Tensor d t -> Tensor d t
multiply = binOp "tf.multiply"

(⊙) :: ∀ (d :: Shape) t. Tensor d t -> Tensor d t -> Tensor d t
(⊙) = multiply

matmul :: Tensor (o ': n ': s) t -> Tensor (m ': o ': s) t -> Tensor (m ': n ': s) t
matmul = binOp "tf.matmul"


sigmoid, tanh, log :: ∀ s. Tensor s 'Float32 -> Tensor s 'Float32
sigmoid = unOp "tf.sigmoid"
tanh = unOp "tf.tanh"
log = unOp "tf.tanh"

split0 :: ∀ m n batchShape t. (KnownNat n, KnownNat m, KnownLen batchShape) =>
          Tensor ((n + m) ': batchShape) t -> Gen (Tensor (n ': batchShape) t, Tensor (m ': batchShape) t)
split0 (T x) = do
  v1 <- newVar
  v2 <- newVar
  gen (v1 <> "," <> v2 <> " = " <> funcall "tf.split" [x, list [show n, show m], "axis=" <> showShapeLen @batchShape])
  return (T v1, T v2)
  where n = natVal (Proxy @ n)
        m = natVal (Proxy @ m)

concat0 :: ∀ ys d1 d2 t. (KnownShape ys) =>  T (d1 ': ys) t -> T (d2 ': ys) t -> T ((d1 + d2) ': ys) t
concat0 t u =
  let T x = t
      T y = u
  in (T (funcall "tf.concat" [brackets (commas [x,y]), "axis=" <> show axis]))
  where axis = shapeLen @ ys -- check

expandDim :: ∀ s0 s t. KnownShape s => Tensor (s0 ++ s) t -> Tensor (s0 ++ (1 ': s)) t
expandDim (T x) = (T (funcall "tf.expand_dims" [x, "axis=" <> show (shapeLen @ s)]))

expandDim0 :: ∀ s t. KnownShape s => Tensor s t -> Tensor ((1 ': s)) t
expandDim0 = expandDim @ '[]


squeeze :: ∀ s0 s1 t. KnownLen s1 => Tensor (s0 ++ (1 ': s1)) t -> Tensor (s0 ++ s1) t
squeeze (T x) = T (funcall "tf.squeeze" [x, "axis=" <> show (shapeLen @ s1)])

squeeze0 :: ∀ s t. KnownLen s => Tensor (1 ': s) t -> Tensor s t
squeeze0 = squeeze @ '[]


unstack :: ∀ s (n::Nat) t. (KnownShape s, KnownNat n) => Tensor (n ': s) t -> Gen (V n (T s t))
unstack (T x) = do
  v <- newVar
  gen (v <> " = " <> funcall "tf.unstack" [x, "axis=" <> show (shapeLen @ s)] )
  return $ V $ [ T $ v <> brackets (show i)| i <- [0..n Prelude.- 1] ]
        where n = natVal (Proxy @ n)

stack :: ∀ s (n::Nat) t. (KnownShape s) => V n (T s t) -> Tensor (n ': s) t
stack (V xs) = T (funcall "tf.stack" [(list [x | T x <- xs]), "axis=" <> show (shapeLen @ s)])

transpose :: ∀ s t. T (Reverse s) t -> T s t
transpose = unOp "tf.transpose"

gather :: ∀s n indexShape t. T (s ++ '[n]) t -> T indexShape 'Int32 -> T (s ++ indexShape) t
gather = binOp "tf.gather"

negate :: ∀ s t. T s t -> T s t
negate = unOp "-"

-------------------------
-- Generic parameters

class Parameter p where
  parameter :: String -> Gen p

instance KnownShape shape => Parameter (T shape t) where
  parameter s = parameter' s zeros

instance (Parameter p, Parameter q) => Parameter (p,q) where
  parameter s = (,) <$> parameter (s<>"_fst") <*> parameter (s<>"_snd")

instance (Parameter p1, Parameter p2, Parameter p3) => Parameter (p1,p2,p3) where
  parameter s = (,,) <$> parameter (s<>"_1") <*> parameter (s<>"_2") <*> parameter (s<>"_3")

instance (Parameter p1, Parameter p2, Parameter p3, Parameter p4) => Parameter (p1,p2,p3,p4) where
  parameter s = (,,,) <$> parameter (s<>"_1") <*> parameter (s<>"_2") <*> parameter (s<>"_3") <*> parameter (s<>"_4")

--------------------
-- "Contrib"


----------------
-- Helpers
matvecmulBatch :: ∀ s cols rows t. (KnownNat cols, KnownShape s) =>  Tensor (cols ': rows ': s) t -> Tensor (cols ': s) t -> Tensor (rows ': s) t
matvecmulBatch m v = squeeze0 (matmul m (expandDim0 v))

matvecmul :: Tensor (cols ': rows ': '[]) t -> Tensor (cols ': batchSize ': '[]) t -> Tensor (rows ': batchSize ': '[]) t
matvecmul m v = matmul v (transpose m)

(∙) :: Tensor '[cols, rows] t -> Tensor '[cols,batchSize] t -> Tensor '[rows,batchSize] t 
(∙) = matvecmul


---------------------
-- Linear functions


-- A linear function form a to b is a matrix and a bias.
type (a ⊸ b) = (Tensor '[a,b] 'Float32, Tensor '[b] 'Float32)

-- | Apply a linear function
(#) :: (a ⊸ b) -> T '[a,batchSize] 'Float32 -> Tensor '[b,batchSize] 'Float32
(weightMatrix, bias) # v = weightMatrix ∙ v ⊕ bias

-----------------------
-- Feed-forward layers

-- | embedding layer
embedding :: ∀ embeddingSize numObjects batchSize t. Tensor '[numObjects, embeddingSize] t -> Tensor '[1,batchSize] 'Int32 -> Tensor '[embeddingSize,batchSize] t
embedding param input = gather @ '[embeddingSize] (transpose param) (squeeze0 input)

dense :: (n ⊸ m) -> Tensor '[n, batchSize] 'Float32 -> Tensor '[m, batchSize] 'Float32
dense lf t = (lf # t)

softmax0 :: T (n ': s) 'Float32 -> T (n ': s) 'Float32
softmax0 = unOp "tf.nn.softmax"

-------------------------------
-- Loss functions

-- type Loss s bs = Tensor (s++'[bs]) -> Tensor (s++'[bs]) -> Tensor '[bs]
type Loss s bs t = Last s ~ bs => Tensor s t -> Tensor s t -> Tensor '[bs] 'Float32


crossEntropy :: Tensor '[n,bs] 'Float32 -> Tensor '[n,bs] 'Float32 -> Tensor '[bs] 'Float32
crossEntropy y_ y = negate (reduceSum0 (y_ ⊙ log y))


-------------------------------
-- RNN layers and combinators

type RnnCell state input output = (state , input) -> Gen (state , output)


-- | Any pure function (feed-forward layer) can be transformed into a
-- cell by ignoring the RNN state.
-- timeDistribute :: (Tensor (a ': batchShape) -> Tensor (b ': batchShape)) -> RnnCell () (a ': batchShape) (b ': batchShape)
-- timeDistribute pureLayer (s,a) = return (s, pureLayer a)

timeDistribute :: (Tensor a t -> Tensor b t') -> RnnCell () (T a t) (T b t')
timeDistribute pureLayer (s,a) = return (s, pureLayer a)

lstm :: ∀ n x bs. (KnownNat bs) => 
        (((n + x) ⊸ n),
         ((n + x) ⊸ n),
         ((n + x) ⊸ n),
         ((n + x) ⊸ n)) ->
        RnnCell (T '[n,bs] 'Float32, T '[n,bs] 'Float32) (Tensor '[x,bs] 'Float32) (Tensor '[n,bs] 'Float32)
lstm (wf,wi,wc,wo) ((ht1 , ct1) , input) = do
  c <- assign ((f ⊙ ct1) ⊕ (i ⊙ cTilda))
  h <- assign (o ⊕ tanh c)
  return ((c , h) , h)
  where  hx :: T '[ n + x, bs ] 'Float32
         hx = concat0 ht1 input
         f = sigmoid (wf # hx)
         i = sigmoid (wi # hx)
         cTilda = tanh (wc # hx)
         o = sigmoid (wo # hx)

-- | Stack two RNN cells
stackLayers :: RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0,s1) a c
stackLayers l1 l2 ((s0,s1),x) = do
  (s0',y) <- l1 (s0,x)
  (s1',z) <- l2 (s1,y)
  return ((s0',s1'),z)

infixr .--.
(.--.) :: forall s0 a b s1 c. RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0, s1) a c
(.--.) = stackLayers

-- | @addAttention attn l@ adds the attention function @attn@ to the
-- layer @l@.  Note that @attn@ can depend in particular on a constant
-- external value @h@ which is the complete input to pay attention to.
-- The type parameter @x@ is the size of the portion of @h@ that the
-- layer @l@ will observe.
addAttention :: KnownShape s => (state -> T (x ': s) t) -> RnnCell state (T ((a+x) ': s) t) (T (b ': s) t) -> RnnCell state (T (a ': s) t) (T (b ': s) t)
addAttention attn l (s,a) = l (s,concat0 a (attn s))


-- | Build a RNN by repeating a cell @n@ times.
rnn :: ∀ n state input output t u.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell state (T input t) (T output u) ->
       (state , Tensor (n ': input) t) -> Gen (state , Tensor (n ': output) u)
rnn cell (s0, t) = do
  xs <- unstack t
  (sFin,us) <- chain cell (s0,xs)
  return (sFin,stack us)

-- TODO: attempt to do this with
-- tf.foldl

-- | RNN helper
chain :: ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b)
chain _ (s0 , V []) = return (s0 , V [])
chain f (s0 , V (x:xs)) = do
  (s1,x') <- f (s0 , x)
  (sFin,V xs') <- chain f (s1 , V xs)
  return (sFin,V (x':xs'))




example1 :: KnownNat batchSize => Tensor '[20,1,batchSize] 'Int32 -> Gen (Tensor '[20,batchSize] 'Float32)
example1 input = do
  (embs,lstm1,lstm2,w) <- parameter "params"
  (_sFi,out) <- rnn (timeDistribute (embedding @ 50 @ 100000 embs)
                     .--.
                     (lstm @ 150 lstm1)
                     .--.
                     (lstm @ 150 lstm2)
                     .--.
                     timeDistribute (sigmoid . squeeze0 . dense  w))
                (() |> (zeros,zeros) |> (zeros,zeros) |> (),input)
  return out

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>

type Batch s batchSize = Tensor (s++'[batchSize])


compile :: ∀ bs s (s'::Shape) t u. (Last s'~bs, KnownShape s, KnownShape s',KnownTyp t, KnownTyp u) =>
          (Tensor s t -> Gen (Tensor s' u)) -> Loss s' bs u -> Gen ()
compile model lossFunction = do
  x <- placeholder "x"
  y_ <- placeholder "y_"
  y <- assign =<< model x
  "y" <-- y
  "loss" <-- reduceMeanAll (lossFunction y y_)
  return ()


generate :: Gen () -> String
generate s = unlines (fromGen s (\() _v0 -> []) 0)

main :: IO ()
main = putStrLn $ generate $ compile @1024 example1 crossEntropy

{-> main

x = tf.placeholder(tf.int32, shape=[1024,1,20])
var0 = tf.Variable(tf.zeros([50,100000]), name="params_1")
var1 = tf.Variable(tf.zeros([150,200]), name="params_2_1_fst")
var2 = tf.Variable(tf.zeros([150]), name="params_2_1_snd")
var3 = tf.Variable(tf.zeros([150,200]), name="params_2_2_fst")
var4 = tf.Variable(tf.zeros([150]), name="params_2_2_snd")
var5 = tf.Variable(tf.zeros([150,200]), name="params_2_3_fst")
var6 = tf.Variable(tf.zeros([150]), name="params_2_3_snd")
var7 = tf.Variable(tf.zeros([150,200]), name="params_2_4_fst")
var8 = tf.Variable(tf.zeros([150]), name="params_2_4_snd")
var9 = tf.Variable(tf.zeros([150,300]), name="params_3_1_fst")
var10 = tf.Variable(tf.zeros([150]), name="params_3_1_snd")
var11 = tf.Variable(tf.zeros([150,300]), name="params_3_2_fst")
var12 = tf.Variable(tf.zeros([150]), name="params_3_2_snd")
var13 = tf.Variable(tf.zeros([150,300]), name="params_3_3_fst")
var14 = tf.Variable(tf.zeros([150]), name="params_3_3_snd")
var15 = tf.Variable(tf.zeros([150,300]), name="params_3_4_fst")
var16 = tf.Variable(tf.zeros([150]), name="params_3_4_snd")
var17 = tf.Variable(tf.zeros([150,150]), name="params_4_fst")
var18 = tf.Variable(tf.zeros([150]), name="params_4_snd")
var19 = tf.unstack(x, axis=2)
var20 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), tf.gather(tf.transpose(var0), tf.squeeze(var19[0], axis=1))], axis=1), tf.transpose(var1)), var2)), tf.zeros([1024,150])), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), tf.gather(tf.transpose(var0), tf.squeeze(var19[0], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), tf.gather(tf.transpose(var0), tf.squeeze(var19[0], axis=1))], axis=1), tf.transpose(var5)), var6))))
var21 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), tf.gather(tf.transpose(var0), tf.squeeze(var19[0], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var20))
var22 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), var21], axis=1), tf.transpose(var9)), var10)), tf.zeros([1024,150])), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), var21], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), var21], axis=1), tf.transpose(var13)), var14))))
var23 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([tf.zeros([1024,150]), var21], axis=1), tf.transpose(var15)), var16)), tf.tanh(var22))
var24 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var20, tf.gather(tf.transpose(var0), tf.squeeze(var19[1], axis=1))], axis=1), tf.transpose(var1)), var2)), var21), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var20, tf.gather(tf.transpose(var0), tf.squeeze(var19[1], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var20, tf.gather(tf.transpose(var0), tf.squeeze(var19[1], axis=1))], axis=1), tf.transpose(var5)), var6))))
var25 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var20, tf.gather(tf.transpose(var0), tf.squeeze(var19[1], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var24))
var26 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var22, var25], axis=1), tf.transpose(var9)), var10)), var23), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var22, var25], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var22, var25], axis=1), tf.transpose(var13)), var14))))
var27 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var22, var25], axis=1), tf.transpose(var15)), var16)), tf.tanh(var26))
var28 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var24, tf.gather(tf.transpose(var0), tf.squeeze(var19[2], axis=1))], axis=1), tf.transpose(var1)), var2)), var25), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var24, tf.gather(tf.transpose(var0), tf.squeeze(var19[2], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var24, tf.gather(tf.transpose(var0), tf.squeeze(var19[2], axis=1))], axis=1), tf.transpose(var5)), var6))))
var29 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var24, tf.gather(tf.transpose(var0), tf.squeeze(var19[2], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var28))
var30 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var26, var29], axis=1), tf.transpose(var9)), var10)), var27), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var26, var29], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var26, var29], axis=1), tf.transpose(var13)), var14))))
var31 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var26, var29], axis=1), tf.transpose(var15)), var16)), tf.tanh(var30))
var32 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var28, tf.gather(tf.transpose(var0), tf.squeeze(var19[3], axis=1))], axis=1), tf.transpose(var1)), var2)), var29), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var28, tf.gather(tf.transpose(var0), tf.squeeze(var19[3], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var28, tf.gather(tf.transpose(var0), tf.squeeze(var19[3], axis=1))], axis=1), tf.transpose(var5)), var6))))
var33 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var28, tf.gather(tf.transpose(var0), tf.squeeze(var19[3], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var32))
var34 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var30, var33], axis=1), tf.transpose(var9)), var10)), var31), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var30, var33], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var30, var33], axis=1), tf.transpose(var13)), var14))))
var35 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var30, var33], axis=1), tf.transpose(var15)), var16)), tf.tanh(var34))
var36 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var32, tf.gather(tf.transpose(var0), tf.squeeze(var19[4], axis=1))], axis=1), tf.transpose(var1)), var2)), var33), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var32, tf.gather(tf.transpose(var0), tf.squeeze(var19[4], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var32, tf.gather(tf.transpose(var0), tf.squeeze(var19[4], axis=1))], axis=1), tf.transpose(var5)), var6))))
var37 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var32, tf.gather(tf.transpose(var0), tf.squeeze(var19[4], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var36))
var38 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var34, var37], axis=1), tf.transpose(var9)), var10)), var35), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var34, var37], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var34, var37], axis=1), tf.transpose(var13)), var14))))
var39 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var34, var37], axis=1), tf.transpose(var15)), var16)), tf.tanh(var38))
var40 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var36, tf.gather(tf.transpose(var0), tf.squeeze(var19[5], axis=1))], axis=1), tf.transpose(var1)), var2)), var37), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var36, tf.gather(tf.transpose(var0), tf.squeeze(var19[5], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var36, tf.gather(tf.transpose(var0), tf.squeeze(var19[5], axis=1))], axis=1), tf.transpose(var5)), var6))))
var41 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var36, tf.gather(tf.transpose(var0), tf.squeeze(var19[5], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var40))
var42 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var38, var41], axis=1), tf.transpose(var9)), var10)), var39), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var38, var41], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var38, var41], axis=1), tf.transpose(var13)), var14))))
var43 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var38, var41], axis=1), tf.transpose(var15)), var16)), tf.tanh(var42))
var44 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var40, tf.gather(tf.transpose(var0), tf.squeeze(var19[6], axis=1))], axis=1), tf.transpose(var1)), var2)), var41), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var40, tf.gather(tf.transpose(var0), tf.squeeze(var19[6], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var40, tf.gather(tf.transpose(var0), tf.squeeze(var19[6], axis=1))], axis=1), tf.transpose(var5)), var6))))
var45 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var40, tf.gather(tf.transpose(var0), tf.squeeze(var19[6], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var44))
var46 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var42, var45], axis=1), tf.transpose(var9)), var10)), var43), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var42, var45], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var42, var45], axis=1), tf.transpose(var13)), var14))))
var47 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var42, var45], axis=1), tf.transpose(var15)), var16)), tf.tanh(var46))
var48 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var44, tf.gather(tf.transpose(var0), tf.squeeze(var19[7], axis=1))], axis=1), tf.transpose(var1)), var2)), var45), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var44, tf.gather(tf.transpose(var0), tf.squeeze(var19[7], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var44, tf.gather(tf.transpose(var0), tf.squeeze(var19[7], axis=1))], axis=1), tf.transpose(var5)), var6))))
var49 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var44, tf.gather(tf.transpose(var0), tf.squeeze(var19[7], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var48))
var50 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var46, var49], axis=1), tf.transpose(var9)), var10)), var47), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var46, var49], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var46, var49], axis=1), tf.transpose(var13)), var14))))
var51 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var46, var49], axis=1), tf.transpose(var15)), var16)), tf.tanh(var50))
var52 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var48, tf.gather(tf.transpose(var0), tf.squeeze(var19[8], axis=1))], axis=1), tf.transpose(var1)), var2)), var49), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var48, tf.gather(tf.transpose(var0), tf.squeeze(var19[8], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var48, tf.gather(tf.transpose(var0), tf.squeeze(var19[8], axis=1))], axis=1), tf.transpose(var5)), var6))))
var53 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var48, tf.gather(tf.transpose(var0), tf.squeeze(var19[8], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var52))
var54 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var50, var53], axis=1), tf.transpose(var9)), var10)), var51), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var50, var53], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var50, var53], axis=1), tf.transpose(var13)), var14))))
var55 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var50, var53], axis=1), tf.transpose(var15)), var16)), tf.tanh(var54))
var56 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var52, tf.gather(tf.transpose(var0), tf.squeeze(var19[9], axis=1))], axis=1), tf.transpose(var1)), var2)), var53), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var52, tf.gather(tf.transpose(var0), tf.squeeze(var19[9], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var52, tf.gather(tf.transpose(var0), tf.squeeze(var19[9], axis=1))], axis=1), tf.transpose(var5)), var6))))
var57 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var52, tf.gather(tf.transpose(var0), tf.squeeze(var19[9], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var56))
var58 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var54, var57], axis=1), tf.transpose(var9)), var10)), var55), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var54, var57], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var54, var57], axis=1), tf.transpose(var13)), var14))))
var59 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var54, var57], axis=1), tf.transpose(var15)), var16)), tf.tanh(var58))
var60 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var56, tf.gather(tf.transpose(var0), tf.squeeze(var19[10], axis=1))], axis=1), tf.transpose(var1)), var2)), var57), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var56, tf.gather(tf.transpose(var0), tf.squeeze(var19[10], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var56, tf.gather(tf.transpose(var0), tf.squeeze(var19[10], axis=1))], axis=1), tf.transpose(var5)), var6))))
var61 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var56, tf.gather(tf.transpose(var0), tf.squeeze(var19[10], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var60))
var62 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var58, var61], axis=1), tf.transpose(var9)), var10)), var59), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var58, var61], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var58, var61], axis=1), tf.transpose(var13)), var14))))
var63 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var58, var61], axis=1), tf.transpose(var15)), var16)), tf.tanh(var62))
var64 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var60, tf.gather(tf.transpose(var0), tf.squeeze(var19[11], axis=1))], axis=1), tf.transpose(var1)), var2)), var61), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var60, tf.gather(tf.transpose(var0), tf.squeeze(var19[11], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var60, tf.gather(tf.transpose(var0), tf.squeeze(var19[11], axis=1))], axis=1), tf.transpose(var5)), var6))))
var65 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var60, tf.gather(tf.transpose(var0), tf.squeeze(var19[11], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var64))
var66 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var62, var65], axis=1), tf.transpose(var9)), var10)), var63), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var62, var65], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var62, var65], axis=1), tf.transpose(var13)), var14))))
var67 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var62, var65], axis=1), tf.transpose(var15)), var16)), tf.tanh(var66))
var68 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var64, tf.gather(tf.transpose(var0), tf.squeeze(var19[12], axis=1))], axis=1), tf.transpose(var1)), var2)), var65), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var64, tf.gather(tf.transpose(var0), tf.squeeze(var19[12], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var64, tf.gather(tf.transpose(var0), tf.squeeze(var19[12], axis=1))], axis=1), tf.transpose(var5)), var6))))
var69 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var64, tf.gather(tf.transpose(var0), tf.squeeze(var19[12], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var68))
var70 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var66, var69], axis=1), tf.transpose(var9)), var10)), var67), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var66, var69], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var66, var69], axis=1), tf.transpose(var13)), var14))))
var71 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var66, var69], axis=1), tf.transpose(var15)), var16)), tf.tanh(var70))
var72 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var68, tf.gather(tf.transpose(var0), tf.squeeze(var19[13], axis=1))], axis=1), tf.transpose(var1)), var2)), var69), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var68, tf.gather(tf.transpose(var0), tf.squeeze(var19[13], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var68, tf.gather(tf.transpose(var0), tf.squeeze(var19[13], axis=1))], axis=1), tf.transpose(var5)), var6))))
var73 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var68, tf.gather(tf.transpose(var0), tf.squeeze(var19[13], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var72))
var74 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var70, var73], axis=1), tf.transpose(var9)), var10)), var71), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var70, var73], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var70, var73], axis=1), tf.transpose(var13)), var14))))
var75 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var70, var73], axis=1), tf.transpose(var15)), var16)), tf.tanh(var74))
var76 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var72, tf.gather(tf.transpose(var0), tf.squeeze(var19[14], axis=1))], axis=1), tf.transpose(var1)), var2)), var73), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var72, tf.gather(tf.transpose(var0), tf.squeeze(var19[14], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var72, tf.gather(tf.transpose(var0), tf.squeeze(var19[14], axis=1))], axis=1), tf.transpose(var5)), var6))))
var77 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var72, tf.gather(tf.transpose(var0), tf.squeeze(var19[14], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var76))
var78 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var74, var77], axis=1), tf.transpose(var9)), var10)), var75), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var74, var77], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var74, var77], axis=1), tf.transpose(var13)), var14))))
var79 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var74, var77], axis=1), tf.transpose(var15)), var16)), tf.tanh(var78))
var80 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var76, tf.gather(tf.transpose(var0), tf.squeeze(var19[15], axis=1))], axis=1), tf.transpose(var1)), var2)), var77), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var76, tf.gather(tf.transpose(var0), tf.squeeze(var19[15], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var76, tf.gather(tf.transpose(var0), tf.squeeze(var19[15], axis=1))], axis=1), tf.transpose(var5)), var6))))
var81 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var76, tf.gather(tf.transpose(var0), tf.squeeze(var19[15], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var80))
var82 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var78, var81], axis=1), tf.transpose(var9)), var10)), var79), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var78, var81], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var78, var81], axis=1), tf.transpose(var13)), var14))))
var83 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var78, var81], axis=1), tf.transpose(var15)), var16)), tf.tanh(var82))
var84 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var80, tf.gather(tf.transpose(var0), tf.squeeze(var19[16], axis=1))], axis=1), tf.transpose(var1)), var2)), var81), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var80, tf.gather(tf.transpose(var0), tf.squeeze(var19[16], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var80, tf.gather(tf.transpose(var0), tf.squeeze(var19[16], axis=1))], axis=1), tf.transpose(var5)), var6))))
var85 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var80, tf.gather(tf.transpose(var0), tf.squeeze(var19[16], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var84))
var86 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var82, var85], axis=1), tf.transpose(var9)), var10)), var83), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var82, var85], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var82, var85], axis=1), tf.transpose(var13)), var14))))
var87 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var82, var85], axis=1), tf.transpose(var15)), var16)), tf.tanh(var86))
var88 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var84, tf.gather(tf.transpose(var0), tf.squeeze(var19[17], axis=1))], axis=1), tf.transpose(var1)), var2)), var85), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var84, tf.gather(tf.transpose(var0), tf.squeeze(var19[17], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var84, tf.gather(tf.transpose(var0), tf.squeeze(var19[17], axis=1))], axis=1), tf.transpose(var5)), var6))))
var89 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var84, tf.gather(tf.transpose(var0), tf.squeeze(var19[17], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var88))
var90 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var86, var89], axis=1), tf.transpose(var9)), var10)), var87), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var86, var89], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var86, var89], axis=1), tf.transpose(var13)), var14))))
var91 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var86, var89], axis=1), tf.transpose(var15)), var16)), tf.tanh(var90))
var92 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var88, tf.gather(tf.transpose(var0), tf.squeeze(var19[18], axis=1))], axis=1), tf.transpose(var1)), var2)), var89), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var88, tf.gather(tf.transpose(var0), tf.squeeze(var19[18], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var88, tf.gather(tf.transpose(var0), tf.squeeze(var19[18], axis=1))], axis=1), tf.transpose(var5)), var6))))
var93 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var88, tf.gather(tf.transpose(var0), tf.squeeze(var19[18], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var92))
var94 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var90, var93], axis=1), tf.transpose(var9)), var10)), var91), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var90, var93], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var90, var93], axis=1), tf.transpose(var13)), var14))))
var95 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var90, var93], axis=1), tf.transpose(var15)), var16)), tf.tanh(var94))
var96 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var92, tf.gather(tf.transpose(var0), tf.squeeze(var19[19], axis=1))], axis=1), tf.transpose(var1)), var2)), var93), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var92, tf.gather(tf.transpose(var0), tf.squeeze(var19[19], axis=1))], axis=1), tf.transpose(var3)), var4)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var92, tf.gather(tf.transpose(var0), tf.squeeze(var19[19], axis=1))], axis=1), tf.transpose(var5)), var6))))
var97 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var92, tf.gather(tf.transpose(var0), tf.squeeze(var19[19], axis=1))], axis=1), tf.transpose(var7)), var8)), tf.tanh(var96))
var98 = tf.add_n(tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var94, var97], axis=1), tf.transpose(var9)), var10)), var95), tf.multiply(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var94, var97], axis=1), tf.transpose(var11)), var12)), tf.tanh(tf.add_n(tf.matmul(tf.concat([var94, var97], axis=1), tf.transpose(var13)), var14))))
var99 = tf.add_n(tf.sigmoid(tf.add_n(tf.matmul(tf.concat([var94, var97], axis=1), tf.transpose(var15)), var16)), tf.tanh(var98))
y = tf.stack([tf.nn.softmax(tf.add_n(tf.matmul(var23, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var27, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var31, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var35, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var39, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var43, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var47, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var51, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var55, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var59, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var63, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var67, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var71, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var75, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var79, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var83, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var87, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var91, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var95, tf.transpose(var17)), var18)), tf.nn.softmax(tf.add_n(tf.matmul(var99, tf.transpose(var17)), var18))], axis=2)
-}

