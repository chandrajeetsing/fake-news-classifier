# main.py
from AIAgent.pipeline.agent_pipeline import AgentPipeline
from AIAgent.logging import logger

if __name__ == "__main__":
    try:
        logger.info("Execution has started.")
        pipeline = AgentPipeline()
        pipeline.run()
        logger.info("Execution has finished successfully.")
    except Exception as e:
        logger.exception(e) # logger.exception automatically logs the full traceback
        raise e