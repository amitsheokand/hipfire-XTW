// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen prefix-cache channel split + jinja splice tests (no GPU).

use hipfire_generate::common::{
    asst_turn_fingerprint, flatten_qwen_cached_turn_tokens, qwen_build_cached_assistant_turn,
    qwen_lookup_cached_assistant_turn, split_qwen_think_body_tokens, split_qwen_tool_body_tokens,
};
use hipfire_generate::qwen::plan_from_rendered;
use hipfire_loader::AsstTurnCache;
use hipfire_runtime::prompt_frame::{
    build_cached_history_jinja, CachedAssistantBody, CachedAssistantToolBody, CachedAssistantTurn,
    JinjaChatFrame, Message, Role, ToolCall,
};
use hipfire_runtime::tokenizer::Tokenizer;
use serde_json::json;

fn byte_to_gpt2_char(b: u8) -> char {
    if b == b' ' {
        'Ġ'
    } else if (33..=126).contains(&b) {
        char::from(b)
    } else {
        char::from(b)
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn test_tokenizer() -> Tokenizer {
    let mut entries: Vec<String> = Vec::new();
    entries.push(r#""<|im_start|>": 0"#.to_string());
    entries.push(r#""<|im_end|>": 1"#.to_string());
    entries.push(r#""<think>": 2"#.to_string());
    entries.push(r#""</think>": 3"#.to_string());
    entries.push(r#""\n": 7"#.to_string());
    for b in 0u32..=255u32 {
        let ch = byte_to_gpt2_char(b as u8);
        let escaped = json_escape(&ch.to_string());
        entries.push(format!(r#""{}": {}"#, escaped, 100 + b));
    }
    let vocab_block = entries.join(", ");
    let json = format!(
        r#"{{
            "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
            "added_tokens": [
                {{"id": 0, "content": "<|im_start|>", "special": true}},
                {{"id": 1, "content": "<|im_end|>", "special": true}},
                {{"id": 2, "content": "<think>", "special": true}},
                {{"id": 3, "content": "</think>", "special": true}}
            ]
        }}"#,
        vocab = vocab_block,
    );
    Tokenizer::from_hf_json(&json).expect("test tokenizer")
}

#[test]
fn think_body_split_separates_reasoning_and_content_tokens() {
    let tok = test_tokenizer();
    let body = tok.encode("reason</think>ok");
    let (reasoning, remainder) = split_qwen_think_body_tokens(&body, &tok);
    assert!(reasoning.is_some());
    assert!(!remainder.is_empty());
    assert_eq!(
        flatten_qwen_cached_turn_tokens(&qwen_build_cached_assistant_turn(
            body.clone(),
            "reason",
            "ok",
            &[],
            &tok,
        )),
        body
    );
}

#[test]
fn tool_turn_stores_tools_slot_for_jinja_splice() {
    let tok = test_tokenizer();
    let tool_xml = "\n<tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>";
    let body = tok.encode(tool_xml);
    let tool_calls = vec![ToolCall {
        id: Some("call_0".to_string()),
        name: "bash".to_string(),
        arguments: json!({"command": "echo hi"}),
        rendered_body: None,
    }];
    let turn = qwen_build_cached_assistant_turn(body.clone(), "", "", &tool_calls, &tok);
    assert!(turn.content.is_none());
    assert_eq!(turn.tools.len(), 1);
    assert_eq!(turn.tools[0].recipient, "bash");
    assert!(!turn.tools[0].token_ids.is_empty());
}

#[test]
fn jinja_splice_hits_with_reasoning_content_and_tool_turn() {
    let tok = test_tokenizer();
    let template = r#"{% for m in messages %}{% if m.role == 'assistant' %}[A]{% if m.reasoning_content %}<R>{{ m.reasoning_content }}</R>{% endif %}{% if m.tool_calls %}{% for tc in m.tool_calls %}<T n="{{ tc.name }}">{% if tc.rendered_body %}{{ tc.rendered_body }}{% else %}{{ tc.arguments | tojson }}{% endif %}</T>{% endfor %}{% else %}<C>{{ m.content }}</C>{% endif %}[AEND]{% else %}[{{ m.role }}:{{ m.content }}]{% endif %}{% endfor %}{% if add_generation_prompt %}[GEN]{% endif %}"#;
    let frame = JinjaChatFrame {
        tokenizer: &tok,
        template,
        system: None,
        user: "",
        enable_thinking: true,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };

    let reasoning = "plan";
    let tool_body = "echo hi";
    let tool_xml = format!(
        "\n<tool_call>\n<function=bash>\n<parameter=command>\n{tool_body}\n</parameter>\n</function>\n</tool_call>"
    );
    let body = tok.encode(&format!("{reasoning}</think>{tool_xml}"));
    let tool_calls = vec![ToolCall {
        id: Some("call_0".to_string()),
        name: "bash".to_string(),
        arguments: json!({"command": "echo hi"}),
        rendered_body: None,
    }];
    let stored = qwen_build_cached_assistant_turn(body.clone(), reasoning, "", &tool_calls, &tok);
    let fp = asst_turn_fingerprint("", &tool_calls);

    let mut cache = AsstTurnCache::new_from_env();
    cache.insert(fp, stored);

    let prior = tok.encode("[user:run tool]");
    let messages = vec![
        Message {
            role: Role::User,
            content: "run tool".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        },
        Message {
            role: Role::Assistant,
            content: String::new(),
            reasoning_content: Some(reasoning.to_string()),
            name: None,
            rendered_name: None,
            tool_calls: tool_calls.clone(),
            tool_call_id: None,
            tool_plan: String::new(),
        },
        Message {
            role: Role::User,
            content: "next".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        },
    ];

    let primer = tok.encode("<think>\n");
    let rendered = build_cached_history_jinja(&frame, &messages, None, |msg| {
        if msg.role != Role::Assistant {
            return None;
        }
        let normalized = "";
        let fp = asst_turn_fingerprint(normalized, &msg.tool_calls);
        qwen_lookup_cached_assistant_turn(&cache, fp, &primer)
    })
    .expect("jinja splice");

    let plan = plan_from_rendered(&prior, rendered, true, &[], false, "test");
    assert!(plan.cache_hit, "expected LCP hit on tool+reasoning turn");
    assert_eq!(plan.start_pos, prior.len());
}

#[test]
fn lookup_replay_prepends_primer_to_reasoning_channel() {
    let tok = test_tokenizer();
    let reasoning_ids = tok.encode("r");
    let turn = CachedAssistantTurn {
        reasoning: Some(CachedAssistantBody {
            token_ids: reasoning_ids,
            text: "r".to_string(),
        }),
        tools: vec![],
        content: Some(CachedAssistantBody {
            token_ids: tok.encode("ok"),
            text: "ok".to_string(),
        }),
    };
    let mut cache = AsstTurnCache::new_from_env();
    let fp = asst_turn_fingerprint("ok", &[]);
    cache.insert(fp, turn);

    let primer = tok.encode("<think>\n");
    let replay = qwen_lookup_cached_assistant_turn(&cache, fp, &primer).expect("hit");
    let mut expected = primer.clone();
    expected.extend_from_slice(&tok.encode("r"));
    assert_eq!(replay.reasoning.as_ref().unwrap().token_ids, expected);
}
