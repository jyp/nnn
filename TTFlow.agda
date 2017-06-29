module TTFlow where

open import Data.List
open import Data.Unit
open import Data.Nat
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
matmul _ _ = MKT {!!}


{-

mul : ∀ {batchShape} -> Tensor (batchShape) -> Tensor (batchShape) -> Tensor (batchShape)
mul _ _ = MKT

flatten : ∀ {xs} -> Tensor xs -> Tensor [ product xs ]
flatten _ = MKT

eye : (numRows numColumns : ℕ) → ∀{batchShape} -> Tensor ( numColumns ∷ numRows ∷ batchShape  )
eye numRows numColumns = MKT

scalar_mul : Tensor [] -> ∀ {d} -> Tensor d -> Tensor d
scalar_mul _ _ = MKT

split : ∀{xs ys d1 d2} -> Tensor (xs ++ (d1 + d2) ∷ ys) -> Tensor (xs ++ d1 ∷ ys) × Tensor (xs ++ d2 ∷ ys)
split _ = MKT , MKT

split0 : ∀ n -> ∀{ys m} -> Tensor ((n + m) ∷ ys) -> Tensor (n ∷ ys) × Tensor (m ∷ ys)
split0 n _ = MKT , MKT

transpose : ∀ {xs} -> Tensor xs -> Tensor (reverse xs)
transpose _ = MKT

diag : (n : ℕ) → Tensor ( n ∷ n ∷ [] )
diag n = MKT


-- ------------------------------------
-- Higher-level things (not in TF)

last : ∀{xs ys d} -> Tensor (xs ++ d ∷ ys) -> Tensor (xs ++ ys)
last _ = MKT

concat0 : ∀{ys d1 d2} -> Tensor (d1 ∷ ys) -> Tensor (d2 ∷ ys) -> Tensor (d1 + d2 ∷ ys) 
concat0 _ _ = MKT

concat1 : ∀{xs ys d1 d2} -> Tensor (xs ∷ d1 ∷ ys) -> Tensor (xs ∷ d2 ∷ ys) -> Tensor (xs ∷ (d1 + d2) ∷ ys) 
concat1 _ _ = MKT


rnn : ∀ {state input output} n ->
         ((Tensor state × Tensor input) -> (Tensor state × Tensor output)) ->
         (Tensor state × Tensor (n ∷ input)) -> (Tensor state × Tensor (n ∷ output))
rnn n cell _ = MKT , MKT

sigmoid : ∀ {d} -> Tensor d -> Tensor d
sigmoid _ = MKT

tanh : ∀ {d} -> Tensor d -> Tensor d
tanh _ = MKT


matvecmul : ∀ {batchShape cols rows} -> Tensor (cols ∷ rows ∷ batchShape) -> Tensor (cols ∷ batchShape) -> Tensor (rows ∷ batchShape)
matvecmul _ _ = MKT


lstm : ∀ n {x} -> (Wf : Tensor ((n + x) ∷ n ∷ [])) ->
                  (Wi : Tensor ((n + x) ∷ n ∷ [])) ->
                  (WC : Tensor ((n + x) ∷ n ∷ [])) ->
                  (Wo : Tensor ((n + x) ∷ n ∷ [])) ->
                  Tensor [ n + n  ] × Tensor [ x ] ->
                  Tensor [ n + n ] × Tensor [ n ]
lstm n {x} Wf Wi Wc Wo ( state , input ) with split0 n state
... | ht-1 , Ct-1 = concat0 C h , h
  where  hx : Tensor [ n + x ]
         hx = concat0 ht-1 input
         f = sigmoid (matvecmul Wf hx) -- TODO: biases
         i = sigmoid (matvecmul Wi hx)
         C~ = tanh (matvecmul Wi hx)
         o = sigmoid (matvecmul Wo hx)
         C = add_n (mul f Ct-1)  (mul i C~)
         h = add_n o (tanh C)
-}
