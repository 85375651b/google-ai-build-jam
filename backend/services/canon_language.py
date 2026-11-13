"""Conservative evidence compiler. Unrecognized meaning requires author confirmation."""
import re
from backend.schemas import EntityOperation

REF = r"(?:Planet|Star|Moon|Country|City|Character|Family|Organization|Religion|Language|Event|Species|Institution)\s+[\w'-]+"
NAME = r"[\w'’-]+(?: [A-Z][\w'’-]+)*"


def compile_decisions(decisions, author_text):
    operations = []
    last_name = None
    def add(action, ref, value, quote, **kwargs):
        ref = re.sub(r"['’]s$", '', ref.strip())
        operations.append(EntityOperation(action=action, entity_ref=ref.strip(), value=value,
            source_quote=quote, entity_type=ref.split()[0].lower(), **kwargs))
    for decision in decisions:
        quote = decision.source_quote
        text = re.sub(r"^(?:actually[, ]+|i want )", '', quote.strip(), flags=re.I).rstrip('.!')
        # Name transfer is one batch, with no invented replacement for the old owner.
        transfer = re.search(rf"^({NAME}) is ({REF})'s name,? not ({REF})(?:'s)?$", text, re.I)
        if transfer:
            name, owner, excluded = transfer.groups()
            add('reject_name', excluded, name, quote)
            add('rename', owner, name, quote)
            last_name = name
            continue
        owner = re.search(rf"^(?:that's|that is) ({REF})'s name$", text, re.I)
        if owner and last_name:
            add('rename', owner[1], last_name, quote)
            continue
        negative = re.search(rf"^({REF}) (?:is not|isn't|cannot be|can't be|must not be|must never be)(?: (?:called|named))? ({NAME})$", text, re.I)
        negative2 = re.search(rf"^(?:don't|do not|can't|cannot) (?:use|name) ({NAME}) (?:for|as) ({REF})$", text, re.I)
        negative3 = re.search(rf"^I (?:don't|do not) want ({REF}) to be ({NAME})$", text, re.I)
        if negative or negative2 or negative3:
            ref, name = negative.groups() if negative else negative3.groups() if negative3 else (negative2[2], negative2[1])
            if name[0].isupper() or re.search(r'\b(?:called|named|name)\b', text, re.I):
                add('reject_name', ref, name, quote); last_name = name
                continue
        reserved = re.search(rf"^({NAME}) is (?:reserved for|for) ({REF})$", text, re.I)
        if reserved:
            add('reserve_name', reserved[2], reserved[1], quote); continue
        named = re.search(rf"^({REF})(?:'s name is| is (?:called|named)| should be called| to be called) ({NAME})$", text)
        if not named:
            named = re.search(rf"^(?:change|rename) ({REF})(?:'s name)? to ({NAME})$", text, re.I)
        if not named:
            named = re.search(rf"^({REF}) is ([A-Z][\w'’-]*(?: [A-Z][\w'’-]*)*)$", text)
        if named and not re.search(r'\b(?:and|not|because|has|orbits|contains)\b', named[2], re.I):
            add('rename', named[1], named[2], quote); last_name = named[2]; continue
        alias = re.search(rf"^({REF}) (?:is also known as|has the alias) ({NAME})$", text)
        if alias:
            add('add_alias', alias[1], alias[2], quote); continue
        prop = re.search(rf"^({REF})'s ([a-z_ ]+?) is (.+)$", text, re.I)
        age = re.search(rf'^({REF}) is (\d+(?:\.\d+)?) years old$', text, re.I)
        if prop or age:
            ref, field, value = prop.groups() if prop else (age[1], 'age', age[2])
            if re.match(r'(?:not|maybe|unknown|undecided)\b', value, re.I):
                continue
            if re.fullmatch(r'-?\d+(?:\.\d+)?', value):
                value = float(value) if '.' in value else int(value)
            add('set_property', ref, value, quote, property_key=field.lower().replace(' ', '_')); continue
        parent = re.search(rf'^({REF}) is the biological parent of ({REF})$', text, re.I)
        treaty = re.search(rf'^({REF}) has a treaty with ({REF})$', text, re.I)
        participant = re.search(rf'^({REF}) participates in ({REF})$', text, re.I)
        relation = parent or treaty or participant
        if relation:
            add('add_relationship', relation[1], {}, quote, related_entity_ref=relation[2],
                relationship_type='parent_of' if parent else 'treaty_with' if treaty else 'participates_in')
    # Stable de-duplication, including same-message correction overrides.
    seen, result = set(), []
    for operation in operations:
        signature = (operation.action, operation.entity_ref.casefold(), operation.property_key, str(operation.value), operation.related_entity_ref)
        if signature not in seen:
            seen.add(signature); result.append(operation)
    return result
