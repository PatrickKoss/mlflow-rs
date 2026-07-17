//! Behavioral tests for registry search (plan T7.3), complementing the
//! differential corpus (`search_corpus.rs`, which pins ordering + page
//! boundaries + tokens against the genuine Python store). These target
//! semantics that are awkward to express as a corpus case: workspace isolation,
//! max_results-threshold validation, the `_is_querying_prompt` bypass edge
//! cases, the N+1 pagination boundary + token round-trip, deleted-MV
//! invisibility, MV-search-omits-aliases, order-by tiebreaks, and empty `IN`.
//!
//! Each test gets a fresh [`TempDb`] (SQLite fixture copy, or a live
//! Postgres/MySQL database reset to a clean slate — see
//! `mlflow-test-support`), so the same test bodies run across all three
//! dialects (plan T2.2).

use mlflow_error::ErrorCode;
use mlflow_registry::RegistryStore;
use mlflow_test_support::TempDb;

const WS: &str = "default";
const WS_A: &str = "team-a";
const WS_B: &str = "team-b";
const IS_PROMPT: &str = "mlflow.prompt.is_prompt";

async fn store(temp: &TempDb) -> RegistryStore {
    RegistryStore::new(temp.connect().await)
}

async fn rm_names(
    s: &RegistryStore,
    ws: &str,
    filter: Option<&str>,
    max: i64,
    order: &[String],
) -> Vec<String> {
    let page = s
        .search_registered_models(ws, filter, max, order, None)
        .await
        .expect("search rm");
    page.registered_models.into_iter().map(|r| r.name).collect()
}

async fn mv_ids(
    s: &RegistryStore,
    ws: &str,
    filter: Option<&str>,
    max: i64,
    order: &[String],
) -> Vec<String> {
    let page = s
        .search_model_versions(ws, filter, max, order, None)
        .await
        .expect("search mv");
    page.model_versions
        .into_iter()
        .map(|m| format!("{}/{}", m.name, m.version))
        .collect()
}

// ---------------------------------------------------------------------------
// Workspace isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_is_workspace_scoped() {
    let tmp = TempDb::new("ws_scope").await;
    let s = store(&tmp).await;

    s.create_registered_model(WS_A, "alpha", &[("t", "a")], None)
        .await
        .unwrap();
    s.create_model_version(WS_A, "alpha", "src-a", None, &[], None, None)
        .await
        .unwrap();
    s.create_registered_model(WS_B, "beta", &[("t", "b")], None)
        .await
        .unwrap();
    s.create_model_version(WS_B, "beta", "src-b", None, &[], None, None)
        .await
        .unwrap();

    // Each workspace only sees its own model / version.
    assert_eq!(rm_names(&s, WS_A, None, 100, &[]).await, vec!["alpha"]);
    assert_eq!(rm_names(&s, WS_B, None, 100, &[]).await, vec!["beta"]);
    assert_eq!(mv_ids(&s, WS_A, None, 100, &[]).await, vec!["alpha/1"]);
    assert_eq!(mv_ids(&s, WS_B, None, 100, &[]).await, vec!["beta/1"]);

    // A name filter for the other workspace's model finds nothing.
    assert!(rm_names(&s, WS_A, Some("name = 'beta'"), 100, &[])
        .await
        .is_empty());
    // A tag filter is workspace-scoped too.
    assert_eq!(
        rm_names(&s, WS_A, Some("tags.t = 'a'"), 100, &[]).await,
        vec!["alpha"]
    );
    assert!(rm_names(&s, WS_A, Some("tags.t = 'b'"), 100, &[])
        .await
        .is_empty());
}

// ---------------------------------------------------------------------------
// max_results threshold / validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rm_max_results_threshold_enforced() {
    let tmp = TempDb::new("rm_thresh").await;
    let s = store(&tmp).await;
    let err = s
        .search_registered_models(WS, None, 1001, &[], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err.message.contains("at most 1000"), "{}", err.message);
    // Exactly at the threshold is allowed.
    s.search_registered_models(WS, None, 1000, &[], None)
        .await
        .unwrap();
}

#[tokio::test]
async fn mv_max_results_validation() {
    let tmp = TempDb::new("mv_thresh").await;
    let s = store(&tmp).await;
    // Non-positive rejected.
    let err = s
        .search_model_versions(WS, None, 0, &[], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err.message.contains("positive integer"), "{}", err.message);
    // Above threshold rejected.
    let err = s
        .search_model_versions(WS, None, 200_001, &[], None)
        .await
        .unwrap_err();
    assert!(err.message.contains("at most 200000"), "{}", err.message);
    // At the threshold allowed.
    s.search_model_versions(WS, None, 200_000, &[], None)
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// Prompt inclusion/exclusion incl. untagged-row semantics + bypass
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prompt_exclusion_and_bypass_semantics() {
    let tmp = TempDb::new("prompt").await;
    let s = store(&tmp).await;

    // Two normal models (no prompt tag) + two prompts (is_prompt='true').
    s.create_registered_model(WS, "m1", &[], None)
        .await
        .unwrap();
    s.create_registered_model(WS, "m2", &[], None)
        .await
        .unwrap();
    s.create_registered_model(WS, "p1", &[(IS_PROMPT, "true")], None)
        .await
        .unwrap();
    s.create_registered_model(WS, "p2", &[(IS_PROMPT, "true")], None)
        .await
        .unwrap();

    // Default: prompts excluded.
    assert_eq!(rm_names(&s, WS, None, 100, &[]).await, vec!["m1", "m2"]);

    // `= 'true'` and `!= 'false'` bypass the anti-join → return ONLY prompts
    // (those actually tagged), since the tag HAVING-count join requires the tag.
    assert_eq!(
        rm_names(
            &s,
            WS,
            Some(&format!("tags.`{IS_PROMPT}` = 'true'")),
            100,
            &[]
        )
        .await,
        vec!["p1", "p2"]
    );
    assert_eq!(
        rm_names(
            &s,
            WS,
            Some(&format!("tags.`{IS_PROMPT}` != 'false'")),
            100,
            &[]
        )
        .await,
        vec!["p1", "p2"]
    );

    // `= 'false'` and `!= 'true'` do NOT bypass (anti-join active). Because the
    // untagged normal models lack the tag entirely, and the anti-join only
    // removes prompt-tagged rows, these return the normal models — the
    // untagged-row semantics (`search_utils.py:1304,1499`).
    assert_eq!(
        rm_names(
            &s,
            WS,
            Some(&format!("tags.`{IS_PROMPT}` = 'false'")),
            100,
            &[]
        )
        .await,
        vec!["m1", "m2"]
    );
    assert_eq!(
        rm_names(
            &s,
            WS,
            Some(&format!("tags.`{IS_PROMPT}` != 'true'")),
            100,
            &[]
        )
        .await,
        vec!["m1", "m2"]
    );
}

#[tokio::test]
async fn mv_prompt_tag_is_on_version_table_not_model() {
    let tmp = TempDb::new("mv_prompt").await;
    let s = store(&tmp).await;

    // A registered model tagged as a prompt, whose VERSION has no prompt tag.
    s.create_registered_model(WS, "rp", &[(IS_PROMPT, "true")], None)
        .await
        .unwrap();
    s.create_model_version(WS, "rp", "src", None, &[], None, None)
        .await
        .unwrap();
    // A normal model + version.
    s.create_registered_model(WS, "rn", &[], None)
        .await
        .unwrap();
    s.create_model_version(WS, "rn", "src", None, &[], None, None)
        .await
        .unwrap();

    // MV search's prompt anti-join reads `model_version_tags`, which has no
    // is_prompt tag here, so BOTH versions appear by default (matches Python;
    // verified in the corpus too).
    let ids = mv_ids(&s, WS, None, 100, &[]).await;
    assert!(ids.contains(&"rp/1".to_string()));
    assert!(ids.contains(&"rn/1".to_string()));

    // Tagging the VERSION as a prompt then excludes it by default.
    s.set_model_version_tag(WS, "rp", "1", IS_PROMPT, "true")
        .await
        .unwrap();
    let ids = mv_ids(&s, WS, None, 100, &[]).await;
    assert!(!ids.contains(&"rp/1".to_string()));
    assert!(ids.contains(&"rn/1".to_string()));
}

// ---------------------------------------------------------------------------
// AND-of-tags
// ---------------------------------------------------------------------------

#[tokio::test]
async fn and_of_tags_requires_every_key() {
    let tmp = TempDb::new("and_tags").await;
    let s = store(&tmp).await;
    s.create_registered_model(WS, "a", &[("x", "1"), ("y", "2")], None)
        .await
        .unwrap();
    s.create_registered_model(WS, "b", &[("x", "1")], None)
        .await
        .unwrap();
    s.create_registered_model(WS, "c", &[("y", "2")], None)
        .await
        .unwrap();

    // Only `a` has BOTH x=1 and y=2.
    assert_eq!(
        rm_names(&s, WS, Some("tags.x = '1' AND tags.y = '2'"), 100, &[]).await,
        vec!["a"]
    );
    // Single-key filter matches the two with x=1.
    assert_eq!(
        rm_names(&s, WS, Some("tags.x = '1'"), 100, &[]).await,
        vec!["a", "b"]
    );
}

// ---------------------------------------------------------------------------
// MV attribute aliases + IN + deleted invisibility
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mv_run_id_in_and_source_alias() {
    let tmp = TempDb::new("mv_attr").await;
    let s = store(&tmp).await;
    s.create_registered_model(WS, "m", &[], None).await.unwrap();
    s.create_model_version(WS, "m", "s3://a/1", Some("run_a"), &[], None, None)
        .await
        .unwrap();
    s.create_model_version(WS, "m", "s3://b/2", Some("run_b"), &[], None, None)
        .await
        .unwrap();
    s.create_model_version(WS, "m", "file:///c/3", Some("run_c"), &[], None, None)
        .await
        .unwrap();

    // run_id IN
    let mut ids = mv_ids(&s, WS, Some("run_id IN ('run_a', 'run_c')"), 100, &[]).await;
    ids.sort();
    assert_eq!(ids, vec!["m/1", "m/3"]);

    // source_path alias → `source` column.
    assert_eq!(
        mv_ids(&s, WS, Some("source_path = 's3://b/2'"), 100, &[]).await,
        vec!["m/2"]
    );
    assert_eq!(
        mv_ids(
            &s,
            WS,
            Some("source_path LIKE 's3://%'"),
            100,
            &["version_number ASC".into()]
        )
        .await,
        vec!["m/1", "m/2"]
    );

    // version_number alias → `version` column, numeric comparison.
    assert_eq!(
        mv_ids(&s, WS, Some("version_number > 2"), 100, &[]).await,
        vec!["m/3"]
    );
}

#[tokio::test]
async fn deleted_versions_invisible() {
    let tmp = TempDb::new("mv_deleted").await;
    let s = store(&tmp).await;
    s.create_registered_model(WS, "m", &[], None).await.unwrap();
    s.create_model_version(WS, "m", "src1", None, &[], None, None)
        .await
        .unwrap();
    s.create_model_version(WS, "m", "src2", None, &[], None, None)
        .await
        .unwrap();
    s.delete_model_version(WS, "m", "1").await.unwrap();

    // Deleted version 1 must not appear.
    assert_eq!(mv_ids(&s, WS, None, 100, &[]).await, vec!["m/2"]);
    assert_eq!(
        mv_ids(&s, WS, Some("name = 'm'"), 100, &[]).await,
        vec!["m/2"]
    );
}

#[tokio::test]
async fn mv_search_does_not_populate_aliases() {
    let tmp = TempDb::new("mv_alias").await;
    let s = store(&tmp).await;
    s.create_registered_model(WS, "m", &[], None).await.unwrap();
    s.create_model_version(WS, "m", "src", None, &[], None, None)
        .await
        .unwrap();
    s.set_registered_model_alias(WS, "m", "champion", "1")
        .await
        .unwrap();

    let page = s
        .search_model_versions(WS, None, 100, &[], None)
        .await
        .unwrap();
    let mv = &page.model_versions[0];
    // `search_model_versions` returns entities WITHOUT aliases (Python parity).
    assert!(mv.aliases.is_empty());

    // But `search_registered_models` returns the model with its aliases.
    let rm_page = s
        .search_registered_models(WS, None, 100, &[], None)
        .await
        .unwrap();
    assert_eq!(rm_page.registered_models[0].aliases.len(), 1);
    assert_eq!(rm_page.registered_models[0].aliases[0].alias, "champion");
}

// ---------------------------------------------------------------------------
// order_by tiebreaks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rm_order_by_name_and_tiebreak() {
    let tmp = TempDb::new("rm_order").await;
    let s = store(&tmp).await;
    for n in ["c", "a", "b"] {
        s.create_registered_model(WS, n, &[], None).await.unwrap();
    }
    // Default order is name ASC.
    assert_eq!(rm_names(&s, WS, None, 100, &[]).await, vec!["a", "b", "c"]);
    // Explicit name DESC.
    assert_eq!(
        rm_names(&s, WS, None, 100, &["name DESC".into()]).await,
        vec!["c", "b", "a"]
    );
    // `creation_timestamp` is NOT a valid store order-by key for RM.
    let err = s
        .search_registered_models(WS, None, 100, &["creation_timestamp ASC".into()], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
}

#[tokio::test]
async fn mv_order_by_name_then_version_tiebreak() {
    let tmp = TempDb::new("mv_order").await;
    let s = store(&tmp).await;
    s.create_registered_model(WS, "a", &[], None).await.unwrap();
    s.create_registered_model(WS, "b", &[], None).await.unwrap();
    for _ in 0..3 {
        s.create_model_version(WS, "a", "src", None, &[], None, None)
            .await
            .unwrap();
    }
    s.create_model_version(WS, "b", "src", None, &[], None, None)
        .await
        .unwrap();

    // Default MV order: last_updated DESC, name ASC, version DESC. All same
    // timestamps here (created back-to-back), so name ASC then version DESC wins.
    let ids = mv_ids(&s, WS, None, 100, &[]).await;
    // `a` versions in descending version order, then `b`.
    let pos_a3 = ids.iter().position(|x| x == "a/3").unwrap();
    let pos_a1 = ids.iter().position(|x| x == "a/1").unwrap();
    assert!(pos_a3 < pos_a1, "version DESC tiebreak: {ids:?}");

    // Explicit name ASC, version ASC.
    assert_eq!(
        mv_ids(
            &s,
            WS,
            None,
            100,
            &["name ASC".into(), "version_number ASC".into()]
        )
        .await,
        vec!["a/1", "a/2", "a/3", "b/1"]
    );
}

// ---------------------------------------------------------------------------
// Pagination: N+1 boundary + token round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pagination_boundary_and_token_round_trip() {
    let tmp = TempDb::new("paging").await;
    let s = store(&tmp).await;
    for n in ["a", "b", "c", "d", "e"] {
        s.create_registered_model(WS, n, &[], None).await.unwrap();
    }

    // Page size 2 → pages [a,b],[c,d],[e], with a token after each full page.
    let mut collected: Vec<String> = Vec::new();
    let mut token: Option<String> = None;
    let mut pages = 0;
    loop {
        let page = s
            .search_registered_models(WS, None, 2, &[], token.as_deref())
            .await
            .unwrap();
        collected.extend(page.registered_models.iter().map(|r| r.name.clone()));
        token = page.next_page_token;
        pages += 1;
        if token.is_none() {
            break;
        }
    }
    assert_eq!(collected, vec!["a", "b", "c", "d", "e"]);
    assert_eq!(pages, 3);

    // Exact-multiple boundary: 4 models, page size 2 → the SECOND page is full
    // but there's no third page; Python still emits a token on the full 2nd page
    // (over-fetch of N+1 returns only 2), then the follow-up returns empty & no
    // token. Verify a page whose size == max_results still yields a token when a
    // next row exists, and none when it doesn't.
    let p1 = s
        .search_registered_models(WS, None, 5, &[], None)
        .await
        .unwrap();
    // 5 models, max 5: no over-fetch surplus → no next token.
    assert_eq!(p1.registered_models.len(), 5);
    assert!(p1.next_page_token.is_none());

    let p2 = s
        .search_registered_models(WS, None, 4, &[], None)
        .await
        .unwrap();
    // 5 models, max 4: over-fetch sees a 5th → token present.
    assert_eq!(p2.registered_models.len(), 4);
    assert!(p2.next_page_token.is_some());
    // Following the token yields the last model and no further token.
    let p3 = s
        .search_registered_models(WS, None, 4, &[], p2.next_page_token.as_deref())
        .await
        .unwrap();
    assert_eq!(
        p3.registered_models
            .into_iter()
            .map(|r| r.name)
            .collect::<Vec<_>>(),
        vec!["e"]
    );
    assert!(p3.next_page_token.is_none());
}

// ---------------------------------------------------------------------------
// Invalid-attribute + comparator errors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invalid_attribute_and_comparator_rejected() {
    let tmp = TempDb::new("invalid").await;
    let s = store(&tmp).await;

    // RM: only `name` is a valid attribute — an unknown one is rejected by the
    // parser.
    let err = s
        .search_registered_models(WS, Some("foo = 'x'"), 100, &[], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);

    // MV: a parenthesized IN list is only valid for run_id — the parser rejects
    // it for other attributes (`SearchModelVersionUtils._get_value`).
    let err = s
        .search_model_versions(WS, Some("name IN ('a', 'b')"), 100, &[], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(
        err.message.contains("Only the 'run_id' attribute"),
        "{}",
        err.message
    );

    // MV: an unknown attribute is rejected (store comparator validation is
    // reached for a valid key with a bad comparator, e.g. `version_number LIKE`).
    let err = s
        .search_model_versions(WS, Some("version_number LIKE '1'"), 100, &[], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(
        err.message
            .contains("Invalid comparator for attribute version_number"),
        "{}",
        err.message
    );
}
