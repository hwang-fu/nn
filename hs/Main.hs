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

import Data.List (foldl', maximumBy)
import Data.Ord (comparing)
import Data.Char (chr)
import System.IO (hPutStrLn, stderr)
import Control.Monad (foldM, when)
import Data.Maybe (mapMaybe)
import Text.Read (readMaybe)

-- ============================================================
-- Data Types
-- ============================================================

type Vector = [Double]
type Matrix = [[Double]]

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

-- Dot product of two vectors
-- Example: [1,2,3] `dot` [4,5,6] = 1*4 + 2*5 + 3*6 = 32
dot :: Vector -> Vector -> Double
dot xs ys = foldl' (+) 0 $ zipWith (*) xs ys

-- Matrix-vector multiplication
-- Multiplies each row of the matrix by the vector
matVecMul :: Matrix -> Vector -> Vector
matVecMul xss ys = map (\xs -> xs `dot` ys) xss

-- Element-wise vector addition
-- Example: [1,2,3] `vecAdd` [4,5,6] = [5,7,9]
vecAdd :: Vector -> Vector -> Vector
vecAdd = zipWith (+)

-- Single layer forward pass (linear transformation only)
-- Computes: output = weights x input + bias
-- Activation function applied separately
forwardLayer :: Matrix -> Vector -> Vector -> Vector
forwardLayer weights input bias = (weights `matVecMul` input) `vecAdd` bias

-- Complete forward pass through the nn
-- Input (784) -> Hidden (128, ReLU) -> Output (36, softmax)
-- Returns the probability distribution over 36 classes
forward :: NeuralNetwork -> Vector -> Vector
forward nn input =
    output
  where
    hidden = map relu $ forwardLayer (weightsIH nn) input  (biasH nn)
    output = softmax  $ forwardLayer (weightsHO nn) hidden (biasO nn)

-- Convert class index (0-35) to character
-- 0-25  → 'A'-'Z'
-- 26-35 → '0'-'9'
indexToChar :: Int -> Char
indexToChar i
  | i < 26    = chr (65 + i)      -- 'A' ~ 'Z'
  | otherwise = chr (48 + i - 26) -- '0' ~ '9'

-- Find index and value of maximum element
maxIndex :: [Double] -> (Int, Double)
maxIndex xs = maximumBy (comparing snd) (zip [0..] xs)

-- Make prediction on the input image
-- Returns the most confident class with its probability
predict :: NeuralNetwork -> Vector -> Prediction
predict nn input =
    Prediction c idx confidence
  where
    probabilities     = forward nn input
    (idx, confidence) = maxIndex probabilities
    c                 = indexToChar idx

-- ============================================================
-- Weight Initialization (Xavier/Glorot)
-- ============================================================

-- Linear Congruential Generator for pseudo-random numbers
-- Uses standard LCG parameters: a=1103515245, c=12345, m=2^31
lcg :: Int -> Int
lcg seed = (1103515245 * seed + 12345) `mod` (2^31)

-- Generate infinite stream of random doubles in range [-1, 1]
-- Used for weight initialization with Xavier/Glorot scaling
randomStream :: Int -> [Double]
randomStream seed =
    map normalize $ iterate lcg seed
  where
    maxInt = 2^31 :: Int
    normalize x = (fromIntegral x / fromIntegral maxInt) * 2 - 1

-- Split a list into chunks of size n
chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- Initialize weight matrix using Xavier/Glorot initialization
-- Scale: sqrt(2 / (fan_in + fan_out))
initWeights :: Int -> Int -> Int -> Matrix
initWeights seed inputSz outputSz =
    take outputSz $ chunksOf inputSz scaledRandoms
  where
    scale = sqrt (2.0 / fromIntegral (inputSz + outputSz))
    scaledRandoms = map (\r -> r * scale) (randomStream seed)

-- Initialize the neural network with random weights and zero biases
initNeuralNetwork :: Int -> NeuralNetwork
initNeuralNetwork seed = NeuralNetwork
  { weightsIH = initWeights seed 784 128
  , biasH     = replicate 128 0.0
  , weightsHO = initWeights (seed + 1000) 128 36
  , biasO     = replicate 36 0.0
  }

-- ============================================================
-- Training (Backpropagation with Mini-batch SGD)
-- ===========================================================

-- Transpose matrix (swap rows and columns)
transpose :: Matrix -> Matrix
transpose ([]:_) = []
transpose m = map head m : transpose (map tail m)

-- Outer product of two vectors: creates matrix where m[i][j] = xs[i] * ys[j]
-- Used to compute weight gradients
outerProduct :: Vector -> Vector -> Matrix
outerProduct xs ys = [[x * y | y <- ys] | x <- xs]

-- Train on a single sample using backpropagation
trainSingle :: NeuralNetwork -> Vector -> Int -> Double -> NeuralNetwork
trainSingle nn input targetIdx learningRate = NeuralNetwork
    { weightsIH = newWeightsIH
    , biasH     = newBiasH
    , weightsHO = newWeightsHO
    , biasO     = newBiasO
    }
  where
    -- forward pass
    hidden = map relu (forwardLayer (weightsIH nn) input (biasH nn))
    output = softmax (forwardLayer (weightsHO nn) hidden (biasO nn))
    -- output error
    target       = [
                     if i == targetIdx then 1.0 else 0.0
                   | i <- [0..35]
                   ]
    outputError  = zipWith (-) output target

    -- backpropagate
    transposedHO = transpose (weightsHO nn)
    hiddenError  = zipWith (*)
                     (transposedHO `matVecMul` outputError)
                     (map relu' (forwardLayer (weightsIH nn) input (biasH nn)))

    -- update weights
    newWeightsHO = zipWith (zipWith (\w g -> w - learningRate * g))
                    (weightsHO nn)
                    (outerProduct hidden outputError)

    newBiasO     = zipWith (\b e -> b - learningRate * e)
                     (biasO nn)
                     outputError

    newWeightsIH = zipWith (zipWith (\w g -> w - learningRate * g))
                     (weightsIH nn)
                     (outerProduct input hiddenError)

    newBiasH     = zipWith (\b e -> b - learningRate * e)
                     (biasH nn)
                     hiddenError

-- Train on a batch of samples
trainBatch :: NeuralNetwork -> [TrainingSample] -> Double -> NeuralNetwork
trainBatch nn samples learningRate =
    foldl' trainOne nn samples
  where
    trainOne neuralNetwork sample = trainSingle neuralNetwork (pixels sample) (label sample) learningRate

-- Calculate accuracy rate on a sample set
calcAccuracy :: NeuralNetwork -> [TrainingSample] -> Double
calcAccuracy nn samples =
    fromIntegral correct / fromIntegral (length samples)
  where
    correct = length $ filter isCorrect samples
    isCorrect sample = let pred = predict nn (pixels sample) in predIndex pred == label sample

-- Train with progress reporting every 10 epochs
trainWithProgress :: NeuralNetwork -> [TrainingSample] -> Int -> Double -> IO NeuralNetwork
trainWithProgress nn samples epochs learningRate = do
    hPutStrLn stderr $ "Training on  " ++ show (length samples) ++ " samples"
    hPutStrLn stderr $ "Epochs: " ++ show epochs ++ ", Learning Rate: " ++ show learningRate
    foldM trainEpoch nn [1..epochs]
  where
    trainEpoch neuralNetwork epoch = do
      let trained = trainBatch neuralNetwork samples learningRate
      when (epoch `mod` 10 == 0) $ do
        let accuracy = calcAccuracy trained (take 1000 samples)
        hPutStrLn stderr $ "Epoch " ++ show epoch ++ "/" ++ show epochs ++
                           ", Accuracy: " ++ show (round (accuracy * 100)) ++ "%"
      return trained

-- ============================================================
-- Data Loading (Simple JSON parsing)
-- ============================================================

-- Extract JSON objects from array (simple parser)
extractObjects :: String -> [String]
extractObjects s =
    go 0 "" (dropWhile (/= '{') s)
  where
    go _ acc [] = if null acc then [] else [reverse acc]
    go depth acc (c:cs)
      | c == '{' = go (depth + 1) (c:acc) cs
      | c == '}' = if depth == 1
                   then reverse (c:acc) : go 0 "" (dropWhile (\x -> x /= '{' && x /= ']') cs)
                   else go (depth - 1) (c:acc) cs
      | depth > 0 = go depth (c:acc) cs
      | otherwise = go depth acc cs

-- Parse a single training sample from JSON object
parseSample :: String -> Maybe TrainingSample
parseSample obj = do
  pixels <- extractPixels obj
  label  <- extractLabel obj
  if length pixels == 784
    then Just $ TrainingSample pixels label
    else Nothing

-- Extract pixels array from JSON object
extractPixels :: String -> Maybe Vector
extractPixels obj = do
  let start = findSubstring "\"pixels\":" obj
  case start of
    Nothing -> Nothing
    Just i -> do
      let rest = drop (i + 9) obj
      let arrayStr = takeWhile (/= ']') (dropWhile (/= '[') rest) ++ "]"
      readMaybe $ filter (/= ' ') arrayStr

-- Extract label from JSON object
extractLabel :: String -> Maybe Int
extractLabel obj = do
  let start = findSubstring "\"label\":" obj
  case start of
    Nothing -> Nothing
    Just i -> do
      let rest = drop (i + 8) obj
      let numStr = takeWhile (\c -> c `elem` "0123456789") (dropWhile (== ' ') rest)
      readMaybe numStr

-- Find substring in string
findSubstring :: String -> String -> Maybe Int
findSubstring needle haystack = go 0 haystack
  where
    len = length needle
    go _ [] = Nothing
    go i s@(_:rest)
      | take len s == needle = Just i
      | otherwise = go (i + 1) rest

-- Parse training data from simple JSON format
-- Format: [{"pixels": [...], "label": N}, ...]
parseTrainingData :: String -> [TrainingSample]
parseTrainingData content = mapMaybe parseSample (extractObjects content)

-- ============================================================
-- File I/O
-- ============================================================

saveWeights :: FilePath -> NeuralNetwork -> IO ()
saveWeights path nn = writeFile path (show nn)

loadWeights :: FilePath -> IO (Maybe NeuralNetwork)
loadWeights path = do
  content <- readFile path
  return $ readMaybe content

loadTrainingData :: FilePath -> IO [TrainingSample]
loadTrainingData path = do
  content <- readFile path
  let samples = parseTrainingData content
  return samples

-- ============================================================
-- Output Formatting
-- ============================================================

formatOutput :: Prediction -> [Prediction] -> String
formatOutput main topPredications = concat
    [ "{"
    , "\"character\":\"" ++ [predChar main] ++ "\","
    , "\"confidence\":" ++ show (predConfidence main) ++ ","
    , "\"index\":" ++ show (predIndex main) ++ ","
    , "\"topN\":[" ++ formatTopN topPredications ++ "]"
    , "}"
    ]
  where
    formatTopN preds = drop 1 $ concatMap formatPred preds
    formatPred p     = ",{\"char\":\"" ++ [predChar p] ++
                       "\",\"confidence\":" ++ show (predConfidence p) ++ "}"



