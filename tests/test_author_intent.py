"""Author authority regressions: exercise production policy and persistence, not an LLM oracle."""
import pytest

from backend.database import SessionLocal
from backend.models import Conversation, Fact, FactRevision, Message, World
from backend.schemas import (
    AuthorDecisionSchema, BibleSectionEditItem, CandidateProposalItem,
    CreativePermissionSchema, PostTurnExtractionSchema, TurnAnalysisSchema,
)
from backend.services.author_intent import clear_author_decisions, extraction_eligible, reconcile_turn
from backend.services.agent_orchestrator import agent_orchestrator
from backend.services.context_assembler import context_assembler
from backend.services.llm_service import llm_service
from backend.services.world_service import world_service
from backend.schemas import WorldCreate


def wrong_research():
    # Reproduce both original mistakes: broad research label + unsupported naming grant.
    return TurnAnalysisSchema(
        primary_intent="FACTUAL_RESEARCH",
        creative_permission=CreativePermissionSchema(
            level="BOUNDED_BRAINSTORMING", allow_invented_proper_nouns=True,
            naming_mode="MAJOR_CONTEXTUAL", explicit_authorization_quote="name",
            allowed_targets=["all entities in the world"],
        ),
    )


@pytest.mark.parametrize("text", [
    "Planet X is called Pallene.", "The capital is New Avalon.",
    "Her birthday is October 12.", "The empire was founded 300 years ago.",
    "I want Planet X to be called Pallene.", "Change Planet X's name to Pallene.",
    "Planet X is not Enceladus.", "Actually, her name is Lianne.",
    "Don't use Pallene for Planet X.", "Enceladus belongs to Planet Y, not Planet X.",
    "Planet X is not called Enceladus. That's Planet Y's name.",
    "Actually, I prefer Pallene. Let's use that.", "The capital is actually on the eastern coast.",
    "Actually you can't use Enceladus because it's the name of the first planet of Roserbe (closest to Myrran) and the location of the capital Akalacia.",
])
def test_author_decisions_survive_wrong_research_label(text):
    result = reconcile_turn(text, wrong_research())
    assert clear_author_decisions(result)
    assert extraction_eligible(result)
    assert "CANON_DECLARATION" in result.secondary_intents
    assert result.creative_permission.authorization_kind == "NONE"
    assert not result.creative_permission.allow_invented_proper_nouns
    assert result.creative_permission.naming_mode == "NONE"


@pytest.mark.parametrize("text", [
    "Could Planet X be called Pallene?", "Maybe I should make her older.",
    "What if the empire had two capitals?", "I'm considering changing the planet's name.",
    "I wonder if Pallene would be a good name.", "Maybe Pallene would work.",
    "I think Planet X is Pallene.", "Planet X might be called Pallene.",
    "If Planet X is Pallene, Planet Y is Enceladus.",
])
def test_exploration_is_not_canon_even_if_model_claims_it_is(text):
    model = wrong_research()
    model.author_decisions = [AuthorDecisionSchema(kind="DECLARATION", source_quote=text)]
    result = reconcile_turn(text, model)
    assert not clear_author_decisions(result)
    assert not extraction_eligible(result)
    assert not result.creative_permission.allow_invented_proper_nouns


@pytest.mark.parametrize("text", [
    "What does Enceladus mean?", "Where does the name Pallene come from?",
    "How do real planetary naming conventions work?", "What does this name mean?",
    "What does the name Pallene mean?", "Could you explain what the name Pallene means?",
])
def test_factual_questions_do_not_authorize_creation_or_canon(text):
    model = wrong_research()
    model.author_decisions = [AuthorDecisionSchema(kind="DECLARATION", source_quote=text)]
    result = reconcile_turn(text, model)
    assert result.primary_intent == "FACTUAL_RESEARCH"
    assert not extraction_eligible(result)
    assert result.creative_permission.authorization_kind == "NONE"


@pytest.mark.parametrize("text", [
    "Don't name it yet.", "I haven't chosen a name.", "That name belongs to another character.",
    "I don't want you inventing names.", "What does this name mean?", "The name is wrong.",
    "Don't give Planet X a name yet.", "That's not what happened.",
    "Suggested names are wrong.", "Naming conventions are complicated.",
    "Give me the meaning of Pallene.", "Help me find the origin of Pallene.",
])
def test_naming_topic_is_never_authorization(text):
    result = reconcile_turn(text, wrong_research())
    assert result.creative_permission.authorization_kind == "NONE"
    assert result.creative_permission.allowed_targets == []
    assert not result.creative_permission.allow_invented_proper_nouns
    assert result.creative_permission.naming_mode == "NONE"


@pytest.mark.parametrize("text", [
    "Give me five possible names for Planet X.", "Help me come up with a name.",
    "Help me think of a name for Planet X.", "Brainstorm some naming conventions.",
    "Can you suggest five names for Planet X?",
])
def test_explicit_suggestions_are_scoped_and_not_author_decisions(text):
    result = reconcile_turn(text, wrong_research())
    assert result.creative_permission.authorization_kind == "SUGGEST"
    assert result.creative_permission.allow_invented_proper_nouns is ("conventions" not in text)
    assert result.creative_permission.allowed_targets == [text]
    assert result.creative_permission.explicit_authorization_quote == text
    assert not clear_author_decisions(result)


@pytest.mark.parametrize("text", [
    "You decide what Planet X should be called.",
    "I trust your judgment on this planet's name.",
    "Choose the best option from those suggestions.",
])
def test_delegation_requires_its_author_clause_and_does_not_expand_scope(text):
    result = reconcile_turn(text, wrong_research())
    assert result.creative_permission.authorization_kind == "DELEGATE"
    assert result.creative_permission.allowed_targets == [text]
    assert result.creative_permission.requested_option_count == 1
    assert not clear_author_decisions(result)


def test_semantic_authorization_supports_non_template_language():
    text = "Would you be willing to dream up an appropriate toponym for Planet X?"
    model = wrong_research()
    model.creative_permission = CreativePermissionSchema(
        level="BOUNDED_BRAINSTORMING", authorization_kind="SUGGEST",
        explicit_authorization_quote=text, allowed_targets=["Planet X's name"],
        allow_invented_proper_nouns=True, resolved_target="Planet X",
    )
    result = reconcile_turn(text, model)
    assert result.creative_permission.authorization_kind == "SUGGEST"
    assert result.creative_permission.allowed_targets == [text]
    # A selected keyword instead of the whole author speech act cannot authorize anything.
    model.creative_permission.explicit_authorization_quote = "toponym"
    assert reconcile_turn(text, model).creative_permission.authorization_kind == "NONE"


def test_mixed_decisions_research_and_suggestions_are_independent():
    text = "I don't want Planet X to be Enceladus. That's Planet Y's name. Help me find a different name for X."
    result = reconcile_turn(text, wrong_research())
    assert len(clear_author_decisions(result)) == 2
    assert result.creative_permission.authorization_kind == "SUGGEST"
    assert result.creative_permission.allowed_targets == ["Help me find a different name for X."]
    mixed = reconcile_turn("Actually, Planet X is Pallene. Can you research the historical origin of that name?", wrong_research())
    assert clear_author_decisions(mixed)
    assert extraction_eligible(mixed)
    assert mixed.creative_permission.authorization_kind == "NONE"


def test_uncertain_other_owner_does_not_discard_clear_prohibition():
    result = reconcile_turn("Don't use Enceladus for Planet X. That's the name I have in mind for another planet.", wrong_research())
    assert [d.source_quote for d in clear_author_decisions(result)] == ["Don't use Enceladus for Planet X."]
    assert result.ambiguity_status == "NEEDS_CLARIFICATION"
    assert result.clarification_question
    assert not result.creative_permission.allow_invented_proper_nouns


def test_clear_decision_and_uncertainty_in_one_sentence():
    result = reconcile_turn("Planet X is Pallene, but maybe Planet Y should be Enceladus.", wrong_research())
    assert [d.source_quote for d in clear_author_decisions(result)] == ["Planet X is Pallene"]
    assert result.uncertain_statements == ["maybe Planet Y should be Enceladus."]


def test_latest_prohibition_revokes_earlier_creation_permission():
    result = reconcile_turn("Give me five names for Planet X. Don't name it yet.", wrong_research())
    assert result.creative_permission.authorization_kind == "NONE"
    assert not result.creative_permission.allow_invented_proper_nouns


@pytest.fixture
def workspace():
    with SessionLocal() as db:
        world = world_service.create_world(db, WorldCreate(name="Author policy regression"))
        conv = db.query(Conversation).filter(Conversation.world_id == world.id).first()
        yield db, world, conv
        world_service.delete_world(db, world)


def run_turn(workspace, monkeypatch, text, extraction, reply="Acknowledged.", classification=None):
    db, world, conv = workspace
    captured = []
    def structured(prompt, response_schema, **kwargs):
        if response_schema is TurnAnalysisSchema:
            return classification or wrong_research()
        captured.append(prompt)
        return extraction
    monkeypatch.setattr(llm_service, "generate_structured", structured)
    result = agent_orchestrator.analyze_turn(db, world, text, conversation=conv)
    user = Message(conversation_id=conv.id, role="USER", content=text)
    assistant = Message(conversation_id=conv.id, role="ASSISTANT", content=reply)
    db.add_all([user, assistant]); db.commit()
    updates = agent_orchestrator.extract_post_turn_proposals(db, world, conv, user, assistant, result)
    return result, updates, captured


def test_actual_reported_bug_is_eligible_and_only_author_decisions_persist(workspace, monkeypatch):
    db, world, conv = workspace
    text = "Actually you can't use Enceladus because it's the name of the first planet of Roserbe (closest to Myrran) and the location of the capital Akalacia."
    db.add_all([
        Message(conversation_id=conv.id, role="USER", content="My world has Cindris's tenth planet and Roserbe's first planet."),
        Message(conversation_id=conv.id, role="ASSISTANT", content="Consider Enceladus for Cindris's tenth planet."),
        Fact(world_id=world.id, title="Unrelated Planet Z", content="Planet Z is named Atlas."),
    ]); db.commit()
    before = db.query(Fact).filter(Fact.world_id == world.id, Fact.title == "Unrelated Planet Z").one().content
    extraction = PostTurnExtractionSchema(proposals=[
        CandidateProposalItem(proposed_title="Roserbe first planet", proposed_content="Roserbe's first planet is Enceladus, closest to Myrran, and contains Akalacia.", source_quote=text),
        CandidateProposalItem(proposed_title="Cindris exclusion", proposed_content="Cindris's tenth planet must not use Enceladus.", source_quote=text),
        CandidateProposalItem(proposed_title="Unauthorized replacement", proposed_content="Cindris's tenth planet is Pallene.", source_quote="name", auto_apply_to_bible=True),
        CandidateProposalItem(proposed_title="Unrelated change", proposed_content="Planet Z is renamed.", source_quote="I authorize Planet Z."),
    ])
    analysis, result, captured = run_turn(workspace, monkeypatch, text, extraction)
    assert extraction_eligible(analysis)
    assert analysis.creative_permission.authorization_kind == "NONE"
    assert captured and text in captured[0]
    facts = db.query(Fact).filter(Fact.world_id == world.id).all()
    assert len(facts) == 3
    assert any("Roserbe's first planet is Enceladus" in f.content for f in facts)
    assert not any("Pallene" in f.content for f in facts)
    assert db.query(Fact).filter(Fact.title == "Unrelated Planet Z").one().content == before


def test_suggestions_cannot_auto_approve_or_rewrite_bible(workspace, monkeypatch):
    text = "Give me five possible names for Planet X."
    extraction = PostTurnExtractionSchema(
        bible_section_updates=[BibleSectionEditItem(section_slug="cosmology-physics", section_title="Cosmology", updated_markdown="Planet X is named Pallene.", source_quote=text)],
        proposals=[CandidateProposalItem(proposed_title="Suggested name", proposed_content="Planet X is named Pallene.", source_quote=text, auto_apply_to_bible=True, proposed_provenance="AUTHOR_CREATED")],
    )
    _, result, _ = run_turn(workspace, monkeypatch, text, extraction)
    assert len(result["staged_proposals"]) == 1
    assert not result["updated_sections"]
    db, world, _ = workspace
    assert db.query(Fact).filter(Fact.world_id == world.id).count() == 0


def test_explicit_approval_persists_without_invention_permission(workspace, monkeypatch):
    text = "I like Pallene. Let's use that for Planet X."
    extraction = PostTurnExtractionSchema(proposals=[CandidateProposalItem(
        proposed_title="Planet X", proposed_content="Planet X is named Pallene.", source_quote="Let's use that for Planet X.",
    )])
    analysis, _, _ = run_turn(workspace, monkeypatch, text, extraction)
    db, world, _ = workspace
    assert not analysis.creative_permission.allow_invented_proper_nouns
    assert db.query(Fact).filter(Fact.world_id == world.id).one().provenance_type == "AUTHOR_CREATED"


@pytest.mark.parametrize("target,chosen,expected", [("Planet X", "Pallene", 1), ("Planet Y", "Pallene", 0), ("Planet X", "Enceladus", 0)])
def test_delegation_scope_choice_and_provenance(workspace, monkeypatch, target, chosen, expected):
    text = "You decide what Planet X should be called."
    extraction = PostTurnExtractionSchema(proposals=[CandidateProposalItem(
        proposed_title=target, proposed_content=f"{target} is named {chosen}.", source_quote=text,
        decision_origin="DELEGATED", decision_target=target,
    )])
    _, _, captured = run_turn(workspace, monkeypatch, text, extraction, reply="Planet X is named Pallene.")
    db, world, _ = workspace
    facts = db.query(Fact).filter(Fact.world_id == world.id).all()
    assert len(facts) == expected
    assert "Planet X is named Pallene." in captured[0]
    if facts:
        assert facts[0].provenance_type == "AI_GENERATED_AUTHOR_APPROVED"
        revision = db.query(FactRevision).filter(FactRevision.fact_id == facts[0].id).one()
        assert revision.changed_by == "AI_AUTHOR_DELEGATED"
        assert text in revision.change_reason and "Planet X" in revision.change_reason


def test_hypothetical_cannot_persist_and_extraction_failure_cannot_promote_ai(workspace, monkeypatch):
    _, _, captured = run_turn(workspace, monkeypatch, "Maybe Pallene would work.", PostTurnExtractionSchema())
    assert not captured
    def fail_extraction(prompt, response_schema, **kwargs):
        if response_schema is TurnAnalysisSchema:
            return wrong_research()
        raise ValueError("bad structured response")
    monkeypatch.setattr(llm_service, "generate_structured", fail_extraction)
    db, world, conv = workspace
    text = "Planet X is called Pallene."
    analysis = agent_orchestrator.analyze_turn(db, world, text)
    user = Message(conversation_id=conv.id, role="USER", content=text)
    assistant = Message(conversation_id=conv.id, role="ASSISTANT", content="## Planet Y\n\nPlanet Y is named Enceladus and has invented lore.")
    db.add_all([user, assistant]); db.commit()
    result = agent_orchestrator.extract_post_turn_proposals(db, world, conv, user, assistant, analysis)
    assert result["updated_sections"] == []
    assert db.query(Fact).filter(Fact.world_id == world.id).count() == 0


def test_per_turn_prohibition_overrides_world_preference(workspace, monkeypatch):
    db, world, _ = workspace
    world.allow_unrequested_proper_nouns = True
    monkeypatch.setattr(llm_service, "generate_structured", lambda **kwargs: wrong_research())
    analysis = agent_orchestrator.analyze_turn(db, world, "Don't name it yet.")
    prompt = context_assembler.assemble_consultant_system_prompt(db, world, analysis, {"passages": []}, {"sources": []})
    assert "Invented Proper Nouns Permitted: False" in prompt
    assert "<layer_9_context_aware_naming_intelligence" not in prompt


@pytest.mark.parametrize("text", [
    "Don't name it yet.", "What does this name mean?", "Could Planet X be called Pallene?",
    "The name is wrong.", "Suggest no names for Planet X.", "Give me the meaning of Pallene.",
    'Dialogue: "Give me names for Planet X."',
])
def test_even_full_model_authorization_cannot_override_negative_or_factual_language(text):
    model = wrong_research()
    model.creative_permission.authorization_kind = "SUGGEST"
    model.creative_permission.explicit_authorization_quote = text
    assert reconcile_turn(text, model).creative_permission.authorization_kind == "NONE"


def test_mixed_research_does_not_suppress_persistence(workspace, monkeypatch):
    text = "Actually, Planet X is Pallene. Can you research the historical origin of that name?"
    extraction = PostTurnExtractionSchema(proposals=[CandidateProposalItem(
        source_quote="Actually, Planet X is Pallene.", proposed_title="Planet X", proposed_content="Planet X is called Pallene.",
    )])
    analysis, _, captured = run_turn(workspace, monkeypatch, text, extraction)
    db, world, _ = workspace
    assert analysis.primary_intent == "FACTUAL_RESEARCH"
    assert "CANON_DECLARATION" in analysis.secondary_intents
    assert captured
    assert db.query(Fact).filter(Fact.world_id == world.id).one().content == "Planet X is called Pallene."


def test_mixed_suggestions_and_author_constraints_have_separate_canon_status(workspace, monkeypatch):
    text = "I don't want Planet X to be Enceladus. That's Planet Y's name. Help me find a different name for X."
    extraction = PostTurnExtractionSchema(proposals=[
        CandidateProposalItem(source_quote="I don't want Planet X to be Enceladus.", proposed_title="X exclusion", proposed_content="Planet X cannot be Enceladus."),
        CandidateProposalItem(source_quote="That's Planet Y's name.", proposed_title="Y name", proposed_content="Planet Y is Enceladus."),
        CandidateProposalItem(source_quote="Help me find a different name for X.", decision_origin="SUGGESTION", proposed_title="X suggestion", proposed_content="Planet X is Pallene.", auto_apply_to_bible=True),
    ])
    _, updates, _ = run_turn(workspace, monkeypatch, text, extraction)
    db, world, _ = workspace
    facts = db.query(Fact).filter(Fact.world_id == world.id).all()
    assert len(facts) == 2
    from backend.models import EntityName, Entity
    excluded = db.query(EntityName).join(Entity).filter(Entity.world_id == world.id,
        Entity.reference_label == 'Planet X', EntityName.kind == 'REJECTED').one()
    assert excluded.name == 'Enceladus' and excluded.source_quote == "I don't want Planet X to be Enceladus."
    assert not any("Pallene" in fact.content for fact in facts)
    assert len(updates["staged_proposals"]) == 1


@pytest.mark.parametrize("name,expected", [("Pallene", 1), ("InventedUnlistedName", 0)])
def test_contextual_selection_only_chooses_from_the_authorized_options(workspace, monkeypatch, name, expected):
    db, world, conv = workspace
    db.add(Message(conversation_id=conv.id, role="ASSISTANT", content="For Planet X, the suggestions are Pallene or Neso.")); db.commit()
    text = "Choose the best option from those suggestions."
    classification = wrong_research()
    classification.creative_permission = CreativePermissionSchema(
        authorization_kind="DELEGATE", level="BOUNDED_BRAINSTORMING", explicit_authorization_quote=text,
        allowed_targets=["Planet X's name"], resolved_target="Planet X",
    )
    extraction = PostTurnExtractionSchema(proposals=[CandidateProposalItem(
        decision_origin="DELEGATED", decision_target="Planet X", source_quote=text,
        proposed_title="Planet X", proposed_content=f"Planet X is named {name}.",
    )])
    analysis, _, _ = run_turn(workspace, monkeypatch, text, extraction, reply=f"Planet X is named {name}.", classification=classification)
    assert analysis.creative_permission.resolved_target == "Planet X"
    assert not analysis.creative_permission.allow_invented_proper_nouns
    assert db.query(Fact).filter(Fact.world_id == world.id).count() == expected


@pytest.mark.parametrize("text,eligible,kind", [
    ("Planet X is called Pallene.", True, "NONE"),
    ("Could Planet X be called Pallene?", False, "NONE"),
    ("Don't name it yet.", False, "NONE"),
])
def test_classifier_failure_has_conservative_independent_fallback(workspace, monkeypatch, text, eligible, kind):
    def fail(*args, **kwargs):
        raise ValueError("classifier unavailable")
    monkeypatch.setattr(llm_service, "generate_structured", fail)
    db, world, _ = workspace
    result = agent_orchestrator.analyze_turn(db, world, text)
    assert extraction_eligible(result) is eligible
    assert result.creative_permission.authorization_kind == kind


@pytest.mark.parametrize("text", [
    "The name is complicated.", "Naming conventions are complicated.",
    "Enceladus is a real-world moon.", "I'm tired.", "Greek names are interesting.",
    "Give me historical facts about Pallene.",
])
def test_unrelated_conversation_and_research_statements_are_not_canon(text):
    result = reconcile_turn(text, wrong_research())
    assert not clear_author_decisions(result)
    assert not extraction_eligible(result)


def test_naming_conventions_do_not_authorize_inventing_planet_names():
    result = reconcile_turn("Brainstorm some naming conventions.", wrong_research())
    assert result.creative_permission.authorization_kind == "SUGGEST"
    assert not result.creative_permission.allow_invented_proper_nouns
    assert result.creative_permission.naming_mode == "NONE"


def test_model_uncertainty_is_not_overridden_by_a_guessed_delegation_target():
    text = "Choose the best option from those suggestions."
    model = wrong_research()
    model.ambiguity_status = "NEEDS_CLARIFICATION"
    model.clarification_question = "Do you mean Planet X's name or Planet Y's name?"
    model.creative_permission = CreativePermissionSchema(
        level="BOUNDED_BRAINSTORMING", authorization_kind="DELEGATE", resolved_target="Planet X",
        explicit_authorization_quote=text, allowed_targets=["Planet X's name"],
    )
    result = reconcile_turn(text, model, "Planet X has suggestions. Planet Y also has suggestions.")
    assert result.ambiguity_status == "NEEDS_CLARIFICATION"
    assert not extraction_eligible(result)


def test_a_guess_about_an_age_does_not_erase_the_decision_to_change_it():
    result = reconcile_turn("I changed my mind. Let's make her older.", wrong_research())
    assert result.author_decisions[0].kind == "CORRECTION"
    assert result.ambiguity_status == "NEEDS_CLARIFICATION"
    assert not extraction_eligible(result)


@pytest.mark.parametrize("text,proposed", [
    ("Planet X is not Enceladus.", "Planet X is named Pallene."),
    ("Planet X is not Enceladus.", "Planet X is named Enceladus."),
    ("Don't use Pallene for Planet X.", "Planet X is called Pallene."),
    ("Planet X is called Pallene.", "Planet Z is named Pallene."),
])
def test_valid_quote_does_not_authorize_a_replacement_or_unrelated_planet(workspace, monkeypatch, text, proposed):
    extraction = PostTurnExtractionSchema(proposals=[CandidateProposalItem(
        proposed_title="Unauthorized assignment", proposed_content=proposed, source_quote=text,
    )])
    run_turn(workspace, monkeypatch, text, extraction)
    db, world, _ = workspace
    facts = db.query(Fact).filter(Fact.world_id == world.id).all()
    # The backend now independently records the explicit positive author decision,
    # even when the model proposes an unrelated entity. Negative clauses invent none.
    assert [f.content for f in facts] == (["Planet X is called Pallene."] if text == "Planet X is called Pallene." else [])


def test_naming_suggestions_for_x_do_not_create_proposals_for_y(workspace, monkeypatch):
    text = "Give me five names for Planet X."
    extraction = PostTurnExtractionSchema(proposals=[CandidateProposalItem(
        proposed_title="Out of scope", proposed_content="Planet Y is named Pallene.", source_quote=text,
    )])
    _, updates, _ = run_turn(workspace, monkeypatch, text, extraction)
    assert not updates["staged_proposals"]


def test_latest_explicit_name_assignment_supersedes_earlier_assignment(workspace, monkeypatch):
    text = "Planet X is called Enceladus. Actually, Planet X is called Pallene."
    extraction = PostTurnExtractionSchema(proposals=[
        CandidateProposalItem(source_quote="Actually, Planet X is called Pallene.", proposed_title="Latest name", proposed_content="Planet X is named Pallene."),
        CandidateProposalItem(source_quote="Planet X is called Enceladus.", proposed_title="Stale name", proposed_content="Planet X is named Enceladus."),
    ])
    analysis, _, _ = run_turn(workspace, monkeypatch, text, extraction)
    assert [d.source_quote for d in clear_author_decisions(analysis)] == ["Actually, Planet X is called Pallene."]
    db, world, _ = workspace
    facts = db.query(Fact).filter(Fact.world_id == world.id).all()
    assert [fact.content for fact in facts] == ["Planet X is called Pallene."]


@pytest.mark.parametrize("text,expected", [
    ("Planet X is called Enceladus. Don't use Enceladus for Planet X.", "Don't use Enceladus for Planet X."),
    ("Don't use Enceladus for Planet X. Planet X is called Enceladus.", "Planet X is called Enceladus."),
])
def test_later_assignment_and_prohibition_override_only_the_same_name(text, expected):
    result = reconcile_turn(text, wrong_research())
    assert [d.source_quote for d in clear_author_decisions(result)] == [expected]


def test_name_and_other_facts_about_same_planet_do_not_supersede_each_other():
    result = reconcile_turn("Planet X is called Pallene. Planet X is volcanically active.", wrong_research())
    assert len(clear_author_decisions(result)) == 2



def test_lowercase_names_still_obey_the_latest_explicit_assignment():
    result = reconcile_turn("planet X is called enceladus. Actually, planet X is called pallene.", wrong_research())
    assert [d.source_quote for d in clear_author_decisions(result)] == ["Actually, planet X is called pallene."]


def test_section_rewrites_cannot_bypass_the_naming_permission_gate(workspace, monkeypatch):
    text = "Planet X is not Enceladus."
    extraction = PostTurnExtractionSchema(bible_section_updates=[BibleSectionEditItem(
        section_slug="cosmology-physics", section_title="Cosmology", source_quote=text,
        updated_markdown="## Planet X\n\n- Planet X is named Pallene.",
    )])
    _, result, _ = run_turn(workspace, monkeypatch, text, extraction)
    assert not result["updated_sections"]
