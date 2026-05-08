type Value = string | number | boolean | null | undefined | { [key: string]: Value } | Value[]; // Possible value types
type Changes = { [key: string]: { before: Value; after: Value } | Changes }; // Recursive changes type
type ValueRecord = Record<string, Value>;

function findDifferences<T extends ValueRecord>(obj1: T, obj2: T): Changes {
    // Helper function to check if a value is an object (excluding arrays)
    function isObject(obj: Value): obj is ValueRecord {
        return obj && typeof obj === 'object' && !Array.isArray(obj);
    }

    // Recursive function to compare two objects and return the differences
    function compareObjects(o1: ValueRecord, o2: ValueRecord): Changes {
        const changes: Changes = {};

        // Iterate over keys in the first object (o1)
        for (const key in o1) {
            if (isObject(o1[key]) && isObject(o2[key])) {
                // Recursively compare nested objects
                const nestedChanges = compareObjects(o1[key], o2[key]);
                if (Object.keys(nestedChanges).length > 0) {
                    changes[key] = nestedChanges; // Add nested changes if any
                }
            } else if (Array.isArray(o1[key]) && Array.isArray(o2[key])) {
                // Compare arrays
                if (o1[key].length !== o2[key].length || o1[key].some((val, index) => val !== o2[key][index])) {
                    changes[key] = { before: o1[key], after: o2[key] };
                }
            } else {
                // Compare primitive values (or any non-object, non-array values)
                if (o1[key] !== o2[key]) {
                    changes[key] = { before: o1[key], after: o2[key] };
                }
            }
        }

        // Iterate over keys in the second object (o2) to detect new keys
        for (const key in o2) {
            if (!(key in o1)) {
                changes[key] = { before: undefined, after: o2[key] };
            }
        }

        return changes; // Return the collected changes
    }

    return compareObjects(obj1, obj2); // Compare the two input objects
}

export default findDifferences;