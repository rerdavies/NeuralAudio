#pragma once

#include "NeuralModel.h"
#include "NeuralModelImpl.h"
#include <NAM/activations.h>
#include <NAM/get_dsp.h>
#include <NAM/dsp.h>
#include <NAM/registry.h>
#include <NAM/slimmable.h>

namespace NeuralAudio
{
	class NAMModel : public NeuralModelImpl
	{
	public:
		NAMModel()
		{
			nam::activations::Activation::enable_fast_tanh();

			slimmableSize = defaultQualityScaleFactor;
		}

		~NAMModel()
		{
			if (namModel)
				namModel.reset();
		}

		EModelLoadMode GetLoadMode() override
		{
			return EModelLoadMode::NAMCore;
		}

		bool LoadFromJson(const nlohmann::json& modelJson)
		{
			if (namModel)
				namModel.reset();

			ReadNAMConfig(modelJson);

			namModel = nam::get_dsp(modelJson);

			auto* slim = dynamic_cast<nam::SlimmableModel*>(namModel.get());

			if (slim != nullptr )
			{
				isSlimmable = true;

                suppressPrewarm = true;	// Hack to work around NAM Core get_dsp() above forcing prewarm
                if (slimmableSize != -1.0f)
                {
				    slim->SetSlimmableSize(slimmableSize);

                } 
                // Parse the modelJson to find model weights.
                this->slimmableWeights.clear();
                if (modelJson.contains("config"))
                {
                    const auto& config = modelJson["config"];
                    if (config.contains("submodels") && config["submodels"].is_array())
                    {
                        for (const auto& submodel : config["submodels"])
                        {
                            if (submodel.contains("max_value"))
                            {
                                this->slimmableWeights.push_back(submodel["max_value"].get<float>());
                            }
                        }
                    }
                }
			}
            
            SetMaxAudioBufferSize(defaultMaxAudioBufferSize);



			return true;
		}

		bool HasQualityScaling() override
		{
			return isSlimmable;
		}

		float GetQualityScaleFactor() override
		{
			return slimmableSize;
		}

		void SetQualityScaleFactor(float scaleFactor) override
		{
			if (HasQualityScaling())
			{
				if (slimmableSize != scaleFactor)
				{
					slimmableSize = scaleFactor;

					if (namModel != nullptr)
					{
						auto* slim = dynamic_cast<nam::SlimmableModel*>(namModel.get());

						slim->SetSlimmableSize(slimmableSize);

					}
				}
			}
		}
        virtual const std::vector<float> GetSlimmableWeights() const {
            return slimmableWeights;
        }

		void SetMaxAudioBufferSize(const int maxSize) override
		{
			namModel->Reset(namModel->GetExpectedSampleRate(), maxSize);
		}

		void Process(float* input, float* output, size_t numSamples) override
		{
			namModel->process(&input, &output, (int)numSamples);
		}

		void Prewarm() override
		{
			if (suppressPrewarm)
			{
				suppressPrewarm = false;
			}
			else
			{			
				namModel->prewarm();
			}
		}

	private:
        std::vector<float> slimmableWeights;
		std::unique_ptr<nam::DSP> namModel = nullptr;
		float slimmableSize = 1.0f;
		bool isSlimmable = false;
		bool suppressPrewarm = false;
	};
}