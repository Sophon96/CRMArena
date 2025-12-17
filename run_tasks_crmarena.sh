#!/bin/bash

# All tasks now use Google Gemini 2.5 Flash Lite
AGENT_MODEL="gemini-2.5-flash-lite-preview-09-2025"
LLM_PROVIDER="openai"

TASKS=(
    "policy_violation_identification"
    # "monthly_trend_analysis"
    "named_entity_disambiguation"
    "best_region_identification"
    "handle_time"
    "knowledge_qa"
    "transfer_count"
    "case_routing"
    "top_issue_identification"
)

STRATEGIES=(
    # "react"
    # "act"
    "tool_call"
)


EVAL_MODE=aided

ORG_TYPE=original
PRIVACY_AWARE_PROMPT=false

for TASK_CATEGORY in "${TASKS[@]}"; do
    for AGENT_STRATEGY in "${STRATEGIES[@]}"; do

        if [ "$INTERACTIVE" = true ]; then
            RESULT_DIR=results/${ORG_TYPE}_interactive/${AGENT_STRATEGY}_${LLM_PROVIDER}
            LOG_DIR=logs/${ORG_TYPE}_interactive/${AGENT_STRATEGY}_${LLM_PROVIDER}
        else
            RESULT_DIR=results/${ORG_TYPE}/${AGENT_STRATEGY}_${LLM_PROVIDER}
            LOG_DIR=logs/${ORG_TYPE}/${AGENT_STRATEGY}_${LLM_PROVIDER}
        fi

        if [ "$PRIVACY_AWARE_PROMPT" = true ]; then
            RESULT_DIR=${RESULT_DIR}_privacy_aware
            LOG_DIR=${LOG_DIR}_privacy_aware
        fi

        echo "LOG_DIR: $LOG_DIR"
        echo "RESULT_DIR: $RESULT_DIR"

        mkdir -p $LOG_DIR
        mkdir -p $RESULT_DIR

        echo "Running task: $TASK_CATEGORY with strategy: $AGENT_STRATEGY and model: $AGENT_MODEL and provider: $LLM_PROVIDER and privacy_aware_prompt: $PRIVACY_AWARE_PROMPT"

        # Construct log file name including the model
        LOG_FILE="${LOG_DIR}/run_${AGENT_MODEL}_${AGENT_STRATEGY}_${TASK_CATEGORY}_${EVAL_MODE}.log"

       
        python -u run_tasks.py \
            --model "$AGENT_MODEL" \
            --task_category "$TASK_CATEGORY" \
            --agent_eval_mode "$EVAL_MODE" \
            --log_dir "$RESULT_DIR" \
            --agent_strategy "$AGENT_STRATEGY" \
            --llm_provider "$LLM_PROVIDER" \
            --reuse_results \
            --privacy_aware_prompt "$PRIVACY_AWARE_PROMPT" \
            --org_type "$ORG_TYPE" > "$LOG_FILE" 2>&1 &
                
        
    done
done
