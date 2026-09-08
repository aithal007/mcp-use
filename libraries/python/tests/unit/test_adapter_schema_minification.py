"""Unit tests for BaseAdapter.fix_schema()'s token-minification behavior.

Covers both the pre-existing JSON-Schema quirk fixes (type-list -> anyOf, bare
enum -> typed string) as regression checks, and the new minification behavior
(dropping `$schema` and redundant auto-generated `title`s).
"""

import copy

import pytest
from jsonschema_pydantic import jsonschema_to_pydantic
from pydantic import ValidationError

from mcp_use.agents.adapters.base import BaseAdapter


class _ConcreteAdapter(BaseAdapter):
    """Minimal concrete adapter so we can instantiate BaseAdapter for testing."""

    def _convert_tool(self, mcp_tool, connector):
        raise NotImplementedError

    def _convert_resource(self, mcp_resource, connector):
        raise NotImplementedError

    def _convert_prompt(self, mcp_prompt, connector):
        raise NotImplementedError


@pytest.fixture
def adapter() -> _ConcreteAdapter:
    return _ConcreteAdapter()


class TestSchemaMeta:
    """Removal of the '$schema' meta-pointer."""

    def test_top_level_schema_key_stripped(self, adapter):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
        }
        result = adapter.fix_schema(schema)

        assert "$schema" not in result
        assert result["type"] == "object"

    def test_nested_schema_key_stripped(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
        }
        result = adapter.fix_schema(schema)

        assert "$schema" not in result["properties"]["address"]


class TestRedundantTitleStripping:
    """Removal of auto-generated titles that just restate the property key."""

    def test_title_matching_key_snake_case_stripped(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "title": "User Id"},
            },
        }
        result = adapter.fix_schema(schema)

        assert "title" not in result["properties"]["user_id"]

    def test_title_matching_key_camel_case_stripped(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "title": "UserId"},
            },
        }
        result = adapter.fix_schema(schema)

        assert "title" not in result["properties"]["user_id"]

    def test_meaningfully_different_title_preserved(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "title": "Primary Account Identifier",
                },
            },
        }
        result = adapter.fix_schema(schema)

        assert result["properties"]["user_id"]["title"] == "Primary Account Identifier"

    def test_idempotent(self, adapter):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "title": "User Id"},
                "notes": {"type": "string", "title": "Extra context about the user"},
            },
        }
        once = adapter.fix_schema(copy.deepcopy(schema))
        twice = adapter.fix_schema(copy.deepcopy(once))

        assert once == twice
        assert "title" not in once["properties"]["user_id"]
        assert once["properties"]["notes"]["title"] == "Extra context about the user"


class TestExistingQuirkFixesRegression:
    """Pre-existing behaviors must be unaffected by the minification changes."""

    def test_type_list_converted_to_any_of(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "nickname": {"type": ["string", "null"]},
            },
        }
        result = adapter.fix_schema(schema)

        prop = result["properties"]["nickname"]
        assert "type" not in prop
        assert prop["anyOf"] == [{"type": "string"}, {"type": "null"}]

    def test_bare_enum_gets_typed_as_string(self, adapter):
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["active", "inactive"]},
            },
        }
        result = adapter.fix_schema(schema)

        prop = result["properties"]["status"]
        assert prop["type"] == "string"
        assert prop["enum"] == ["active", "inactive"]


class TestRoundTripWithPydantic:
    """The minified schema must still produce a working Pydantic model."""

    def test_validates_correct_and_rejects_incorrect_payloads(self, adapter):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "CreateUserInput",
            "properties": {
                "user_id": {"type": "string", "title": "User Id"},
                "age": {"type": "integer", "title": "Age"},
                "nickname": {"type": ["string", "null"], "title": "Nickname"},
                "status": {"enum": ["active", "inactive"]},
                "address": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "title": "Address",
                    "properties": {
                        "city": {"type": "string", "title": "City"},
                    },
                    "required": ["city"],
                },
            },
            "required": ["user_id", "age", "status", "address"],
        }

        fixed = adapter.fix_schema(schema)

        # Minification happened.
        assert "$schema" not in fixed
        assert "title" not in fixed["properties"]["user_id"]
        assert "$schema" not in fixed["properties"]["address"]

        Model = jsonschema_to_pydantic(fixed)

        valid_instance = Model(
            user_id="abc123",
            age=30,
            nickname=None,
            status="active",
            address={"city": "Springfield"},
        )
        assert valid_instance.user_id == "abc123"
        assert valid_instance.age == 30

        with pytest.raises(ValidationError):
            Model(
                user_id="abc123",
                age="not-an-integer",
                status="active",
                address={"city": "Springfield"},
            )

        with pytest.raises(ValidationError):
            # Missing required 'address'.
            Model(user_id="abc123", age=30, status="active")
