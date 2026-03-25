from open_r1.grpo_uvb import GRPOVideoScriptArguments, main
from trl import GRPOConfig, ModelConfig, TrlParser


if __name__ == "__main__":
    parser = TrlParser((GRPOVideoScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
