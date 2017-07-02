module TTFlow where

open import Data.List
import Data.Vec as Vec
open Vec using (Vec)
open import Data.Unit
import Data.Nat
open Data.Nat hiding (_*_)
import Data.Nat.Show as N
open import Data.Product
import Data.String as S
open S hiding (_++_)

_<>_ = S._++_
infixr 20 _<>_

Shape = List ℕ
-- the convension is REVERSE to what tensorflow uses, so that the batch dimensions are at the end of the list.

Effect = ℕ -> String

Gen : ∀ x -> Set
Gen x = (x -> Effect) -> Effect

newVar : Gen String
newVar = λ k n → k ("var" <> N.show n) (suc n)

gen : String -> Gen ⊤
gen s = λ k n → s <> k tt n

data Tensor (d : Shape) : Set where
  MKT : Gen String -> Tensor d


parens : String -> String
parens x = "(" <> x <> ")"

brackets : String -> String
brackets x = "[" <> x <> "]"

commas : List String -> String
commas [] = ""
commas xs = foldr (\x y -> x <> ", " <> y) "" xs

funcall : String -> List String -> String
funcall f args = f <> (parens (commas args))

binOp : ∀ {s1 s2 s3} -> String -> Tensor s1 -> Tensor s2 -> Tensor s3
binOp op (MKT t) (MKT u) = MKT λ k →  t \x -> u \y -> k (funcall op ( x ∷ y ∷ [] ))

unOp : ∀ {s1 s2} -> String -> Tensor s1 -> Tensor s2
unOp op (MKT t) = MKT λ k →  t (\x -> k ("matmul" <> (parens (x))))

add_n : ∀ {d} -> Tensor d -> Tensor d -> Tensor d
add_n (MKT t1) (MKT t2) = MKT λ k →
  newVar \v ->
  t1 \x ->
  t2 \y ->
  gen (v <> " = tf.add_n(" <> x <> "," <> y <> ")\n") \ _ ->
  k v

showShape : Shape -> String
showShape [] = ""
showShape (x ∷ s) = N.show x <> "," <> showShape s 

parameter : String -> ∀ shape -> Tensor shape
parameter name shape = MKT λ k →
  gen (name <> " = tf.Variable(tf.zeros([" <> showShape shape <> "])) ") \ _ ->
  k name

-- weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
--                       name="weights")


matmul : ∀ {batchShape m n o} -> Tensor (o ∷ n ∷ batchShape) -> Tensor (m ∷ o ∷ batchShape) -> Tensor (m ∷ n ∷ batchShape)
matmul (MKT t) (MKT u) = MKT λ k →
  t (\x -> u \y -> k ("matmul" <> (parens (x <> "," <> y))))

_*_ = matmul

mul : ∀ {batchShape} -> Tensor (batchShape) -> Tensor (batchShape) -> Tensor (batchShape)
mul = binOp "multiply"

_⊙_ = mul

flatten : ∀ {xs} -> Tensor xs -> Tensor [ product xs ]
flatten = unOp "flatten"


eye : (numRows numColumns : ℕ) → ∀{batchShape} -> Tensor ( numColumns ∷ numRows ∷ batchShape  )
eye numRows numColumns {batchShape} = MKT \k ->
  k (funcall "tf.eye" (N.show numRows ∷ N.show numColumns ∷ showShape batchShape ∷ []))

scalar_mul : ∀ {d} -> Tensor [] -> Tensor d -> Tensor d
scalar_mul t u = binOp "tf.scalar_mul" t u

-- split0 : ∀ n -> ∀{ys m} -> Tensor ((n + m) ∷ ys) -> Gen (Tensor (n ∷ ys) × Tensor (m ∷ ys))
-- split0 n {ys} t = ?

transpose : ∀ {xs} -> Tensor xs -> Tensor (reverse xs)
transpose = unOp "tf.transpose"

diag : (n : ℕ) → Tensor ( n ∷ n ∷ [] )
diag n = eye n n


-- ------------------------------------
-- Higher-level things (not in TF)

-- last : ∀{xs ys d} -> Tensor (xs ++ d ∷ ys) -> Tensor (xs ++ ys)
-- last _ = MKT {!!}

concat0 : ∀{ys d1 d2} -> Tensor (d1 ∷ ys) -> Tensor (d2 ∷ ys) -> Tensor (d1 + d2 ∷ ys)
concat0 {ys} (MKT t) (MKT u) = MKT λ k →  t \x -> u \y -> k (funcall "concat" (brackets (commas ( x ∷ y ∷ [] )) ∷ "axis=" <> N.show axis ∷ []))
  where axis : ℕ
        axis = length ys -- check

concat1 : ∀{xs ys d1 d2} -> Tensor (xs ∷ d1 ∷ ys) -> Tensor (xs ∷ d2 ∷ ys) -> Tensor (xs ∷ (d1 + d2) ∷ ys)
concat1 {xs} {ys} (MKT t) (MKT u) = MKT λ k →  t \x -> u \y -> k (funcall "concat" (brackets (commas ( x ∷ y ∷ [] )) ∷ "axis=" <> N.show axis ∷ []))
  where axis : ℕ
        axis = length ys -- check


sigmoid : ∀ {d} -> Tensor d -> Tensor d
sigmoid = unOp "sigmoid"

tanh : ∀ {d} -> Tensor d -> Tensor d
tanh = unOp "tanh"

expandDim0 : ∀ {batchShape} -> Tensor batchShape -> Tensor (1 ∷ batchShape)
expandDim0 {batchShape} (MKT t) = MKT (λ k →
  t \ x ->
  k (funcall "expand_dims" ( x ∷  "axis=" <> N.show (length batchShape) ∷ [])))

squeeze0 : ∀ {batchShape} -> Tensor (1 ∷ batchShape) -> Tensor batchShape
squeeze0 {batchShape} (MKT t) = MKT (λ k →
  t \ x ->
  k (funcall "squeeze" ( x ∷  "axis=" <> N.show (length batchShape) ∷ [])))

matvecmul : ∀ {batchShape cols rows} -> Tensor (cols ∷ rows ∷ batchShape) -> Tensor (cols ∷ batchShape) -> Tensor (rows ∷ batchShape)
matvecmul m v = squeeze0 (matmul m (expandDim0 v))


lstm : ∀ n {x} -> (Wf : Tensor ((n + x) ∷ n ∷ [])) ->
                  (Wi : Tensor ((n + x) ∷ n ∷ [])) ->
                  (WC : Tensor ((n + x) ∷ n ∷ [])) ->
                  (Wo : Tensor ((n + x) ∷ n ∷ [])) ->
                  (Tensor [ n ] × Tensor [ n ]) × Tensor [ x ] ->
                  (Tensor [ n ] × Tensor [ n ]) × Tensor [ n ]
lstm n {x} Wf Wi Wc Wo ((ht-1 , Ct-1) , input) = (C , h) , h
  where  hx : Tensor [ n + x ]
         hx = concat0 ht-1 input
         f = sigmoid (matvecmul Wf hx) -- TODO: biases
         i = sigmoid (matvecmul Wi hx)
         C~ = tanh (matvecmul Wi hx)
         o = sigmoid (matvecmul Wo hx)
         C = add_n (mul f Ct-1)  (mul i C~)
         h = add_n o (tanh C)

untensor : ∀ {n xs} -> Tensor (n ∷ xs) -> Vec (Tensor xs) n
untensor = {!!}

mkTensor : ∀ {n xs} ->  Vec (Tensor xs) n -> Tensor (n ∷ xs)
mkTensor = {!!}

chain : {state a b : Set} -> ∀ {n} -> (state × a -> state × b) → (state × Vec a n) -> state × Vec b n
chain f (s0 , Vec.[]) = s0 , Vec.[]
chain f (s0 , x Vec.∷ v) with chain f {! s1 , ? !}
... | _ = {!!}
  where s1,x' = f (s0 , x)
        s1 = proj₁ s1,x'
        x' = proj₂ s1,x'

rnn : ∀ {state : Set} {input output} n ->
         ((state × Tensor input) -> (state × Tensor output)) ->
         (state × Tensor (n ∷ input)) -> (state × Tensor (n ∷ output))
rnn n cell ( st0 , inputs ) = {!!} , (mkTensor {!!})

example : ∀ n x -> ((Tensor [ n ] × Tensor [ n ]) × Tensor (n ∷ x ∷ [])) -> (Tensor [ n ] × Tensor [ n ]) × Tensor (n ∷ n ∷ [])
example n x = rnn n (lstm _ {!!} {!!} {!!} {!!})

