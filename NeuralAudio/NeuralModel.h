#pragma once

#include <filesystem>
#include <istream>
#include "json.hpp"

namespace NeuralAudio
{
	enum EModelLoadMode
	{
		Internal,
		RTNeural,
		NAMCore
	};

	class NeuralModel
	{
	public:
		static NeuralModel* CreateFromFile(std::filesystem::path modelPath);
		static NeuralModel* CreateFromStream(std::basic_istream<char>& stream, std::filesystem::path extension);

		virtual ~NeuralModel()
		{
		}

		static bool SetLSTMLoadMode(EModelLoadMode val)
		{
			if (!SupportsLSTMLoadMode(val))
				return false;

			lstmLoadMode = val;

			return true;
		}

		static bool SetWaveNetLoadMode(EModelLoadMode val)
		{
			if (!SupportsWaveNetLoadMode(val))
				return false;

			wavenetLoadMode = val;

			return true;
		}

		static bool SupportsWaveNetLoadMode(EModelLoadMode mode);
		static bool SupportsLSTMLoadMode(EModelLoadMode mode);

		static void SetAudioInputLevelDBu(float audioDBu)
		{
			audioInputLevelDBu = audioDBu;
		}

		static void SetDefaultMaxAudioBufferSize(int maxSize)
		{
			defaultMaxAudioBufferSize = maxSize;
		}

		virtual EModelLoadMode GetLoadMode()
		{
			return EModelLoadMode::Internal;
		}

		virtual bool IsStatic()
		{
			return false;
		}

		virtual void SetMaxAudioBufferSize(int maxSize)
		{
			(void)maxSize;
		}

		virtual float GetRecommendedInputDBAdjustment()
		{
			return audioInputLevelDBu - modelInputLevelDBu;
		}

		virtual float GetRecommendedOutputDBAdjustment()
		{
			return -18 - modelLoudnessDB;
		}

        virtual bool HasModelInputLevelDBu() const {
            return hasModelInputLevelDBu;
        }
        virtual float GetModelInputLevelDBu() const {
            return modelInputLevelDBu;
        }
        virtual bool HasModelOutputLevelDBu() const {
            return hasModelOutputLevelDBu;
        }
        virtual float GetModelOutputLevelDBu() const {
            return modelOutputLevelDBu;
        }
        virtual bool HasModelLoudnessDB() const {
            return hasModelLoudnessDB;
        }
        virtual bool GetModelLoudnessDB() const {
            return modelLoudnessDB;
        }


		virtual float GetSampleRate()
		{
			return sampleRate;
		}

		virtual void Process(float* input, float* output, size_t numSamples)
		{
			(void)input;
			(void)output;
			(void)numSamples;
		}

		virtual void Prewarm()
		{
		}

	protected:
		void ReadNAMConfig(const nlohmann::json& modelJson);
		void ReadKerasConfig(const nlohmann::json& modelJson);

        bool hasModelInputLevelDBu = false;
		float modelInputLevelDBu = 12;
        bool hasModelOutputLevelDBu = false;
		float modelOutputLevelDBu = 12;


        bool hasModelGainDb = false;
        float modelGainDb = 0.9;

        bool hasModelLoudnessDB = false;
		float modelLoudnessDB = -18;
        
		float sampleRate = 48000;

		inline static float audioInputLevelDBu = 12;
		inline static EModelLoadMode lstmLoadMode = EModelLoadMode::Internal;
		inline static EModelLoadMode wavenetLoadMode = EModelLoadMode::Internal;
		inline static int defaultMaxAudioBufferSize = 128;

		void Prewarm(size_t numSamples, size_t blockSize)
		{
			std::vector<float> input;
			input.resize(blockSize);
			std::fill(input.begin(), input.end(), 0.0f);

			std::vector<float> output;
			output.resize(blockSize);

			for (size_t block = 0; block < (numSamples / blockSize); block++)
			{
				Process(input.data(), output.data(), blockSize);
			}
		}
	};
}