"""
LinkedIn Sales Navigator URL parser.

Parses the custom query format used in Sales Navigator search URLs
and extracts filters into a Python dictionary.
"""

from urllib.parse import urlparse, parse_qs, unquote


def parse_linkedin_query(query_str: str, decode_values: bool = True) -> dict | list | str | int | bool:
    """
    Parse LinkedIn's custom query format into Python objects.

    The format uses:
    - (key:value, key:value) for objects
    - List((item1), (item2)) for arrays
    - key:value for properties

    Args:
        query_str: The query string to parse
        decode_values: Whether to URL decode primitive string values (for double-encoded content)
    """
    query_str = query_str.strip()

    # Handle List(...) - arrays
    if query_str.startswith("List(") and query_str.endswith(")"):
        inner = query_str[5:-1]
        items = split_top_level(inner)
        return [parse_linkedin_query(item, decode_values) for item in items]

    # Handle (...) - objects
    if query_str.startswith("(") and query_str.endswith(")"):
        inner = query_str[1:-1]
        return parse_object(inner, decode_values)

    # Handle primitive values
    if query_str.lower() == "true":
        return True
    if query_str.lower() == "false":
        return False
    if query_str.isdigit():
        return int(query_str)

    # URL decode string values (handles double-encoded content like %2C -> ,)
    if decode_values:
        return unquote(query_str)
    return query_str


def parse_object(obj_str: str, decode_values: bool = True) -> dict:
    """Parse an object string like 'key:value,key2:value2' into a dict."""
    result = {}
    parts = split_top_level(obj_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Find the first colon that's the key:value separator
        colon_idx = find_key_colon(part)
        if colon_idx == -1:
            continue

        key = part[:colon_idx].strip()
        value = part[colon_idx + 1:].strip()
        result[key] = parse_linkedin_query(value, decode_values)

    return result


def find_key_colon(s: str) -> int:
    """Find the colon that separates key from value (not inside nested structures)."""
    depth = 0
    for i, char in enumerate(s):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == ":" and depth == 0:
            return i
    return -1


def split_top_level(s: str) -> list[str]:
    """Split a string by commas, but only at the top level (not inside parentheses)."""
    parts = []
    current = []
    depth = 0

    for char in s:
        if char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current).strip())

    return parts


def extract_filters_from_url(url: str) -> dict:
    """
    Extract search filters from a LinkedIn Sales Navigator URL.

    Args:
        url: The full Sales Navigator search URL

    Returns:
        A dictionary containing the parsed query parameters including filters
    """
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    result = {}

    # Parse the main query parameter
    if "query" in query_params:
        # parse_qs already does one URL decode, so content values still have one layer of encoding
        query_value = query_params["query"][0]
        result["query"] = parse_linkedin_query(query_value)

    # Include other parameters
    if "sessionId" in query_params:
        result["sessionId"] = query_params["sessionId"][0]

    if "viewAllFilters" in query_params:
        result["viewAllFilters"] = query_params["viewAllFilters"][0] == "true"

    return result


def get_filters(url: str) -> list[dict]:
    """
    Convenience function to extract just the filters from a URL.

    Returns a list of filter objects with type and values.
    """
    parsed = extract_filters_from_url(url)

    if "query" in parsed and isinstance(parsed["query"], dict):
        filters = parsed["query"].get("filters", [])
        return filters if isinstance(filters, list) else []

    return []


if __name__ == "__main__":
    import json

    test_url = (
        "https://www.linkedin.com/sales/search/people?query="
        "(recentSearchParam%3A(id%3A5392147268%2CdoLogHistory%3Atrue)%2C"
        "filters%3AList((type%3ACURRENT_TITLE%2Cvalues%3AList((id%3A128%2C"
        "text%3ASenior%2520Associate%2CselectionType%3AINCLUDED)))%2C"
        "(type%3AREGION%2Cvalues%3AList((id%3A103366113%2Ctext%3AVancouver"
        "%252C%2520British%2520Columbia%252C%2520Canada%2CselectionType%3A"
        "INCLUDED)))%2C(type%3AINDUSTRY%2Cvalues%3AList((id%3A50%2Ctext%3A"
        "Architecture%2520and%2520Planning%2CselectionType%3AINCLUDED)))%2C"
        "(type%3ASCHOOL%2Cvalues%3AList((id%3A4855%2Ctext%3AMcGill%2520"
        "University%2CselectionType%3AINCLUDED)))))&sessionId=JSXVDc8wSbiYwZkPm"
        "%2BvdbQ%3D%3D&viewAllFilters=true"
    )

    print("Full parsed result:")
    result = extract_filters_from_url(test_url)
    print(json.dumps(result, indent=2))

    print("\nExtracted filters:")
    filters = get_filters(test_url)
    print(json.dumps(filters, indent=2))
