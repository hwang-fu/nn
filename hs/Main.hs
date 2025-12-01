{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

{-
  Simple Neural Network for Handwriting Recognition
  Written in pure Haskell - no ML libraries!

  Architecture: Input(784) -> Hidden(128, ReLU) -> Output(36, Softmax)
  Recognizes: A-Z (0-25) and 0-9 (26-35)

  Training data: EMNIST dataset (prepared by prepare_data.py)
-}

module Main where

-- ============================================================
-- Data Types
-- ============================================================

type Vector = [] Double
type Matrix = [] [] Double

data NeuralNetwork = NeuralNetwork
  { weightsIH :: !Matrix    -- 784×128 matrix (input -> hidden layer)
  , biasH     :: !Vector    -- 128 bias values for hidden layer
  , weightsHO :: !Matrix    -- 128×36 matrix (hidden -> output layer)
  , biasO     :: !Vector    -- 36 bias values for output layer
  } deriving (Show, Read)

data Prediction = Prediction
  { predChar        :: Char   -- The predicted character ('A', '5', etc.)
  , predIndex       :: Int    -- Internal class index (0 ~ 35)
  , predConfidence  :: Double -- How confident (0.0 ~ 1.0)
  } deriving (Show)

data TrainingSample = TrainingSample
  { pixels :: !Vector   -- 784 pixel values (28x28 flattened, 0.0 ~ 1.0)
  , label  :: !Int      -- correct answer (0 ~ 35)
  } deriving (Show)

-- ============================================================
-- Operations
-- ============================================================

-- ReLU (Rectified Linear Unit) activation function
-- Returns x if positive, 0 otherwise
relu :: Double -> Double
relu x = max 0 x

-- Derivative of ReLU for backpropagation
-- Returns 1 if x > 0, else 0
relu' :: Double -> Double
relu' x = if x > 0 then 1 else 0

-- Softmax activation function for output layer
-- Converts raw scores to probability distribution
softmax :: Vector -> Vector
softmax xs =
    map (\x -> x / sumExp) expXs
  where
    maxX   = maximum xs
    expXs  = map (\x -> exp(x - maxX)) xs
    sumExp = sum expXs


